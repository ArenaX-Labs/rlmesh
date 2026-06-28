# Running Evaluations

A model and an environment connect through three entry points, from most automated to most hands-on:

| Entry point                       | You get                                                                        | Reach for it when                                                                |
| --------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| {func}`~rlmesh.run`               | One call drives whole episodes and returns a typed {class}`~rlmesh.RunResult`. | Scoring a model: leaderboards, sweeps, CI checks.                                |
| {func}`~rlmesh.session`           | A {class}`~rlmesh.Session` you step by hand (`reset` / `predict` / `step`).    | Rendering, custom stop conditions, branching, or mixing your own per-step logic. |
| `Session.reader` / `Session.read` | A read-only, role-addressed view of each raw observation.                      | Inspecting an env, logging canonical roles, or shaping a reward.                 |

All three resolve the adapter from the environment's published contract and the model's spec, so your prediction code always sees its own declared input/output format with no per-env glue. See {doc}`../user-guide/adapters` for how tags and specs match.

## `run()` — the automated rollout

`run()` pumps full episodes to completion and returns a {class}`~rlmesh.RunResult`. Lead with the bound method on a model:

```python
result = model.run(env, seeds=range(100))
print(f"{result.success_rate:.0%} success, mean reward {result.mean_reward:.2f}")
```

`env` may be a local Gymnasium-style env, an {class}`~rlmesh.EnvFactory` (built and tag-stamped, then driven locally), a remote handle such as a `RemoteEnv`, or a bare address string the loop dials:

```python
result = model.run("tcp://127.0.0.1:5555", seeds=range(100))
```

The module-level {func}`~rlmesh.run` is the same loop over an explicit `(model, env)` pair. Its `model` argument is the flexible one: a bare predict callable, a {class}`~rlmesh.Model` subclass class or instance, or a served `RemoteModel` / `SandboxModel` handle. Pass `rlmesh.RANDOM_SAMPLE` for a baseline that samples the action space and ignores observations:

```python
import rlmesh

result = rlmesh.run(my_policy_fn, env, seeds=range(10))   # any obs -> action callable
baseline = rlmesh.run(rlmesh.RANDOM_SAMPLE, env, max_episodes=10)
```

### Arguments

| Argument            | Default | Meaning                                                                                                                                                             |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `seeds`             | `None`  | Per-episode seed sequence; also sets the episode count unless `max_episodes` is given.                                                                              |
| `max_episodes`      | `None`  | Number of episodes to run; overrides the length of `seeds`.                                                                                                         |
| `instruction`       | `None`  | Overrides every {class}`~rlmesh.adapters.Text` input the spec declares, on each step, at its placement in the input tree. No-op if the spec declares no text input. |
| `execution_horizon` | `1`     | Actions executed per predicted chunk; only engages on a chunk corner (see [below](#execution-horizon-end-to-end)).                                                  |
| `close_env`         | `False` | Shut the env down when the run finishes (opt-in).                                                                                                                   |
| `token`             | `""`    | Auth token for a remote env or model.                                                                                                                               |

With neither `seeds` nor `max_episodes`, `run()` does a single episode.

`execution_horizon` is accepted by both the bound methods (`model.run` / `model.session`) and the module-level {func}`~rlmesh.run` / {func}`~rlmesh.session` convenience, which forwards it through.

### The result

{class}`~rlmesh.RunResult` is immutable and aggregates its episodes:

| Member          | Type                        | Meaning                                                         |
| --------------- | --------------------------- | --------------------------------------------------------------- |
| `.episodes`     | `tuple[EpisodeResult, ...]` | One {class}`~rlmesh.EpisodeResult` per episode.                 |
| `.mean_reward`  | `float`                     | Mean total reward across episodes.                              |
| `.success_rate` | `float`                     | Fraction of episodes that **terminated** rather than truncated. |
| `.num_episodes` | `int`                       | Episode count.                                                  |
| `.total_steps`  | `int`                       | Summed steps across episodes.                                   |

Each {class}`~rlmesh.EpisodeResult` carries `index`, `seed`, `steps`, `reward`, `terminated`, and `truncated`:

```python
for ep in result.episodes:
    print(ep.index, ep.seed, ep.steps, ep.reward, ep.terminated)
```

```{caution}
`success_rate` counts episodes whose env signalled a terminal state (`terminated`), not truncation. That equals task success only for envs that define termination *as* success; a time-limit env that truncates on the cap reports those as non-successes.
```

## `session()` — manual, step-by-step control

`session()` hands back a {class}`~rlmesh.Session` you drive yourself. Use it as a context manager so the env connection (and any managed model) closes on exit:

```python
with model.session(env, instruction="put the cup on the plate") as sess:
    obs, info = sess.reset(seed=0)
    while not sess.done:
        action = sess.predict(obs)
        obs, reward, terminated, truncated, info = sess.step(action)
```

The loop primitives mirror Gymnasium, with the adapter folded in:

- `sess.reset(seed=None)` → `(obs, info)`. Begins an episode; ends the previous one (firing `on_episode_end`) and clears adapter state such as the frame-stack buffer.
- `sess.predict(obs)` → `action`. Applies the model's adapter around the model's own predict: the declarative obs transform, host-side frame stacking, any {class}`~rlmesh.adapters.Custom` code, instruction injection into declared text leaves, and chunk replay (one action per call). Returns an env-ready action.
- `sess.step(action)` → `(obs, reward, terminated, truncated, info)`. Applies the action and records reward and termination.
- `sess.done` is `True` once the current episode terminated or truncated.
- `sess.close()` releases the connection, shuts the env down only on the `close_env` opt-in, and fires `on_close`.

Drive multiple episodes by hand when you want to branch on each step — render a frame, apply your own stop condition, or fork the rollout:

```python
sess = model.session(env)
try:
    for seed in range(10):
        obs, info = sess.reset(seed=seed)
        while not sess.done:
            action = sess.predict(obs)
            obs, reward, terminated, truncated, info = sess.step(action)
            if my_should_stop(info):
                break
finally:
    sess.close()
```

The context-manager form above is the idiomatic one; the explicit `try/finally` is the same thing written out when you cannot wrap the whole loop in a `with`. The module-level {func}`~rlmesh.session` accepts the same flexible `model` argument as {func}`~rlmesh.run`, including `rlmesh.RANDOM_SAMPLE`.

`on_episode_end` fires at every episode boundary (the next `reset()`, or `close()` for the last episode), so a stateful model clears its per-episode state identically whether you drive by hand or call `run()`. {meth}`Session.run <rlmesh.Session.run>` pumps whole episodes through these same primitives and is exactly what `model.run(...)` calls under the hood.

## `read` and `reader` — inspect observations by role

`reader` and `read` give a **read-only**, role-addressed view of a raw observation. They reuse the model adapter pipeline pointed at the consumer ({func}`~rlmesh.adapters.resolve_from_contract` plus the obs transform with a no-op action), so they are encoding-agnostic across envs and never mutate the observation.

`sess.reader(*items)` resolves once and returns a callable mapping a raw observation to `{role: value}`:

```python
import rlmesh.adapters as adapt

with model.session(env) as sess:
    read = sess.reader(adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc"), adapt.EEF_POS)
    obs, _ = sess.reset(seed=0)
    while not sess.done:
        view = read(obs)              # {IMAGE_PRIMARY: ..., EEF_POS: ...}
        screen.show(view[adapt.IMAGE_PRIMARY])
        obs, *_ = sess.step(sess.predict(obs))
```

`sess.read(obs, item)` is the one-shot single-role convenience. The underlying reader is cached per item, so calling it every step does not re-resolve:

```python
ee = sess.read(obs, adapt.EEF_POS)
img = sess.read(obs, adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc"))
```

An **item** is one of:

- A bare role constant (`adapt.EEF_POS`, `adapt.IMAGE_PRIMARY`) — kept in the env's native encoding, using the env's own declared layout.
- A model-input leaf that declares the encoding you want — `adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc")`, `adapt.State(adapt.EEF_POS)`. The adapter converts to that form whatever the env stores.

Roles and leaves are the same vocabulary the rest of the adapter system uses; see {doc}`../user-guide/adapters`. The env must publish adapter tags (via an {class}`~rlmesh.EnvFactory` or `rlmesh.adapters.tag(...)`), or the read raises an `AdapterResolutionError` — there are no roles to address otherwise. Values come back in the env's own framework (NumPy for a Gymnasium env, torch for a torch route).

Three things this is for:

- **Debugging an env** — confirm what a camera returns (shape, layout, value range) without threading it through a model.
- **Logging canonical roles** — record `EEF_POS` or the primary image the same way across heterogeneous envs, since the role addresses the quantity, not the env's key.
- **Reward shaping** — compute a shaped term over canonical roles, e.g. `reward - 0.1 * distance(sess.read(obs, adapt.EEF_POS), goal)`.

## Execution horizon, end to end

A policy that emits an action _chunk_ (ACT, diffusion, flow, VLA action heads) defines a `predict_chunk` or `predict_chunk_batch` corner — see the four corners in {doc}`../user-guide/models`. `execution_horizon` tells the rollout how many actions of each predicted chunk to apply before re-planning:

```python
result = model.run(env, seeds=range(50), execution_horizon=8)
```

The runtime owns the replay: it calls the model once, executes the first `execution_horizon` actions of the returned chunk one per env `step`, then calls the model again. One action is applied per step regardless. The model returns its native chunk; the runtime uses a prefix of it (an autoregressive head that declares `execution_horizon` can instead decode exactly that many).

`execution_horizon` only matters when the model defines a chunk corner. Requesting `execution_horizon > 1` on a model with no `predict_chunk` warns and runs un-chunked (one fresh prediction per step), so the default of `1` is always safe.
