# Performance and Scaling

RLMesh keeps the hot path narrow: the wire is framework-neutral bytes, the adapter encodes only the observation keys a model reads, and frame history never crosses the network. Most of what makes an eval fast is structural, chosen when you write the model and serve the env, not tuned afterward. This page covers the knobs that actually exist in the code, what each one changes, and the things that look like knobs but are not. It then covers the mechanics of long-running and streaming evals: readiness, episode accounting, the session lifecycle, and what happens when a connection drops.

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

The adapter encodes only the observation keys the plan actually reads. An env that returns extra keys, or one unencodable key, does not pay for them and does not abort a step over them. This is automatic; there is no flag, and `describe()` shows which keys the plan touches (see {doc}`troubleshooting`).

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
| Client-side retry / reconnect                     | Not done for you. A dropped session raises; re-dial a fresh client (see {doc}`troubleshooting`).                             |
| `predict_concurrency` (server pipelining)         | Present in the Rust serve options but not in the Python `ServeOptions` constructor.                                          |
| Adapter conversions (resize, normalize, encoding) | Correctness transforms the resolver chooses, not performance dials. Shrink the spec to do less work.                         |
| Frame-stack depth                                 | A model capability set by `stack=N`, sized by the model's needs, not a tuning parameter.                                     |

The lifecycle options that `ServeOptions` does expose, `idle_timeout_seconds`, `drain_timeout_seconds`, `close_timeout_seconds`, and `allow_remote_shutdown`, control shutdown behavior rather than throughput; they belong to the session lifecycle covered below.

## Long-running evals

A short eval is one `run` call that returns a {class}`~rlmesh.RunResult`. A longer one (a hundred-seed sweep, a server that stays up across many model connections, an episode you want to watch as it unfolds) needs a way to know the endpoint is ready, an account of how episodes are counted, and an understanding of how a session opens, drives, and closes. For the API shapes of {func}`~rlmesh.run` and {func}`~rlmesh.session` themselves, see {doc}`evaluation`. For the serving side of readiness and health, see {doc}`serving-environments`.

## Readiness before you stream

A long run should wait for the endpoint to be serving rather than racing its startup. RLMesh exposes two machine-readable signals for this, both documented in full under {doc}`serving-environments`:

- The standard `grpc.health.v1` health service. The overall server health (the empty `""` service name) reports `SERVING` once the listener accepts connections. Probe it with any health client before you dial.
- The env-serve CLI's `--ready-fd`, which writes the resolved bind address once the server is up. Useful when the bind port is `0` and you need the chosen port back.

The public Python clients connect once when you construct them; they do not poll a not-yet-bound endpoint for you. Gate your run on the health signal, then dial.

## Episode accounting

`run` counts episodes from `seeds` and `max_episodes`, and reports each one in the result.

```python
result = model.run(env, seeds=range(100))
print(result.num_episodes, result.total_steps, f"{result.success_rate:.0%}")
```

The count follows three rules:

- `max_episodes` set: run exactly that many. It overrides the length of `seeds`. When both are given and `max_episodes` exceeds the seed count, episodes past the last seed run unseeded.
- only `seeds` set: run one episode per seed, each with its seed.
- neither set: run a single episode.

```python
result = model.run(env, max_episodes=1000)            # 1000 episodes, no seeding
result = model.run(env, seeds=range(50))              # 50 seeded episodes
baseline = rlmesh.run(rlmesh.RANDOM_SAMPLE, env, max_episodes=10)
```

Each episode is an {class}`~rlmesh.EpisodeResult` carrying `index`, `seed`, `steps`, `reward`, `terminated`, `truncated`, and `success`. The `success` field is the env-reported task outcome from the final step's `info` (Gymnasium's `is_success` / `success` key), or `None` when the env emits none. {attr}`RunResult.success_rate <rlmesh.RunResult.success_rate>` prefers that signal and falls back to `terminated` only when it is absent, so a time-limit env should report success through `info` rather than rely on the fallback.

```{caution}
A non-terminating env is bounded: the loop caps each episode at 100,000 steps and
marks the episode `truncated` if it hits the cap, so a broken termination
condition surfaces as a truncation rather than hanging the run forever.
```

```{note}
`run` returns its {class}`~rlmesh.RunResult` only after the last episode finishes.
The public Python API has no per-episode streaming callback. For live progress
during a long run, drive {func}`~rlmesh.session` by hand (below) and read each
episode as it ends, or attach the built-in viewer with `view=` (see {doc}`troubleshooting`).
```

## Session lifecycle

`run` is a loop over the same `reset` / `predict` / `step` primitives a {class}`~rlmesh.Session` exposes. Driving the session yourself is how you stream: inspect each step, apply a custom stop condition, or log per episode as it completes.

```python
with model.session(env, instruction="put the cup on the plate") as sess:
    for seed in range(100):
        obs, info = sess.reset(seed=seed)
        total, steps = 0.0, 0
        while not sess.done:
            obs, reward, terminated, truncated, info = sess.step(sess.predict(obs))
            total += reward
            steps += 1
        print(seed, steps, total, terminated)   # log this episode before the next
```

The lifecycle has a few load-bearing edges.

The env connection opens lazily on the first `reset`, not at construction, so building a session is cheap and the dial happens when you start. `reset` ends the previous episode before starting the next, which fires a stateful model's `on_episode_end` and clears adapter state such as the frame-stack buffer. `predict` applies the model's adapter around its own prediction and replays one action of a chunk per call. `step` records reward and termination, and `sess.done` is `True` once the episode terminated or truncated. `close` releases the connection, fires `on_close`, and shuts the env down only on the `close_env` opt-in.

```{mermaid}
flowchart TD
    A[session built, not connected] --> B[reset: lazy connect, end prev episode, clear adapter state]
    B --> C[predict, then step]
    C -->|not done| C
    C -->|done| D{more episodes?}
    D -->|yes| B
    D -->|no| E[close: on_close, release connection, shut env if close_env]
```

`on_episode_end` fires at every episode boundary, the next `reset` or `close` for the last episode, so a stateful model clears per-episode state identically whether you drive by hand or call `run`. Using the session as a context manager guarantees `close` runs even when a step raises. The explicit `try`/`finally` form is the same thing written out when you cannot wrap the whole loop in a `with`.

## Disconnect and reconnect

The loop does not retry a step, and the public clients do not reconnect a dropped session. When the env or a served model fails mid-episode, the call raises, the session runs its cleanup (end the open episode, `on_close`, `close`), and the exception propagates out of `run`. The in-progress episode is discarded; `run` raises rather than returning the episodes finished so far. The full path and the exception types are in {doc}`troubleshooting`.

To recover a long run, catch the transport error, construct a fresh client, and start a new run from where you left off (your own loop owns the seed cursor). RLMesh classifies a still-binding server as transient internally, but reconnection of an established session is yours to drive.

## Keeping a server alive, and stopping it cleanly

A server that serves many connections over a long window is governed by {class}`~rlmesh.ServeOptions`. Pass it to {class}`~rlmesh.EnvServer` (or a model's `serve`).

```python
import rlmesh

options = rlmesh.ServeOptions(
    idle_timeout_seconds=300,      # stop after 5 min with no activity
    drain_timeout_seconds=30,      # bound draining in-flight requests on shutdown
    close_timeout_seconds=10,      # bound the env close hook on shutdown
    allow_remote_shutdown=False,   # ignore a client shutdown RPC
    token="s3cret",                # require this bearer token on every request
)
rlmesh.EnvServer(env, "0.0.0.0:5555", options=options).serve()
```

Each field controls one part of the lifecycle:

- `idle_timeout_seconds` stops the server after that much inactivity. The window arms when the server starts and every request resets it, so an active eval keeps the server alive and an abandoned one shuts itself down. `None` (the default) never times out.
- `drain_timeout_seconds` bounds how long shutdown waits for in-flight requests to finish. `None` waits indefinitely.
- `close_timeout_seconds` bounds how long the env's close hook may take on shutdown. `None` waits indefinitely.
- `allow_remote_shutdown` decides whether a client `shutdown` RPC is honored. Off by default, so a connected peer cannot stop your server.
- `token` requires a bearer token on every request. `None` or an empty string disables authentication.

The defaults are conservative: no idle shutdown, unbounded drain and close, remote shutdown off, no token. Set them when a server outlives a single run.

## Where next

- {doc}`models`: the four prediction corners and how the runtime fuses, splits, and derives them.
- {doc}`evaluation`: {func}`~rlmesh.run`, {func}`~rlmesh.session`, `execution_horizon`, the chunk-replay loop, and the {class}`~rlmesh.RunResult` fields in full.
- {doc}`serving-environments`: `framework=`, `device=`, the vector serve path, the health service, the ready file descriptor, and bind addresses.
- {doc}`remote-clients`: scalar and vector clients and the connection surface.
- {doc}`troubleshooting`: what a mid-run disconnect or model crash raises, and how to recover.
- {doc}`adapters`: what the spec encodes and why a leaner spec moves less data.
- {doc}`troubleshooting`: the `view=` live viewer and the `read` / `reader` inspection path for watching a run.
