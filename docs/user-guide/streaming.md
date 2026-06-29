# Streaming and long-running evals

A short eval is one `run` call that returns a {class}`~rlmesh.RunResult`. A long one, a hundred-seed sweep, a server that stays up across many model connections, an episode you want to watch as it unfolds, needs a little more: a way to know the endpoint is ready, a clear account of how episodes are counted, and an understanding of how a session opens, drives, and closes. This page covers those mechanics and what happens when a connection drops part-way through.

For the API shapes of {func}`~rlmesh.run` and {func}`~rlmesh.session` themselves, see {doc}`evaluation`. For the serving side of readiness and health, see {doc}`serving-environments`.

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
episode as it ends, or attach the built-in viewer with `view=` (see {doc}`debugging`).
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

The loop does not retry a step, and the public clients do not reconnect a dropped session. When the env or a served model fails mid-episode, the call raises, the session runs its cleanup (end the open episode, `on_close`, `close`), and the exception propagates out of `run`. The in-progress episode is discarded; `run` raises rather than returning the episodes finished so far. The full path and the exception types are in {doc}`error-handling`.

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

- {doc}`evaluation` — {func}`~rlmesh.run`, {func}`~rlmesh.session`, and the {class}`~rlmesh.RunResult` fields in full.
- {doc}`serving-environments` — the health service, the ready file descriptor, and bind addresses.
- {doc}`error-handling` — what a mid-run disconnect or model crash raises, and how to recover.
- {doc}`remote-clients` — connecting to scalar and vector endpoints.
- {doc}`debugging` — the `view=` live viewer and the `read` / `reader` inspection path for watching a run.
