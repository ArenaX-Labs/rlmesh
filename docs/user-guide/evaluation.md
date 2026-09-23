# Running Evaluations

An evaluation drives a model against an environment one step at a time: reset, predict, step, repeat until the episode ends. RLMesh exposes three entry points onto that loop, from one fully automated call down to a read-only peek at a single observation. All three resolve the model-to-environment adapter from the environment's published contract and the model's {class}`~rlmesh.adapters.ModelSpec`, so your prediction code only ever sees its own declared input and output format with no per-environment glue.

This page is the home for the loop itself: {func}`~rlmesh.run`, {func}`~rlmesh.session`, `Session.read` / `Session.reader`, and the execution horizon. See {doc}`adapters` for how tags and specs match into the adapter every entry point applies, and {doc}`models` for the predict corners a model implements.

## Pick an entry point

| Entry point                       | You get                                                                        | Reach for it when                                                                |
| --------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| {func}`~rlmesh.run`               | One call drives whole episodes and returns a typed {class}`~rlmesh.RunResult`. | Scoring a model: leaderboards, sweeps, CI checks.                                |
| {func}`~rlmesh.session`           | A {class}`~rlmesh.Session` you step by hand (`reset` / `predict` / `step`).    | Rendering, custom stop conditions, branching, or mixing your own per-step logic. |
| `Session.reader` / `Session.read` | A read-only, role-addressed view of each raw observation.                      | Inspecting an env, logging canonical roles, or shaping a reward.                 |

The three share one resolution step and diverge only in how much of the loop they drive for you.

```{mermaid}
flowchart LR
    contract["env contract<br/>(published tags)"] --> resolve["resolve adapter"]
    spec["model spec"] --> resolve
    resolve --> run["run()<br/>whole episodes"]
    resolve --> session["session()<br/>step by hand"]
    resolve --> read["read() / reader()<br/>inspect by role"]
```

## `run()`: the automated rollout

`run()` pumps full episodes to completion and returns a {class}`~rlmesh.RunResult`. Lead with the bound method on a model:

```python
result = model.run(env, seeds=range(100))
print(result.success_rate, f"mean reward {result.mean_reward:.2f}")
```

`env` may be a local Gymnasium-style env, an {class}`~rlmesh.EnvFactory` (built and tag-stamped, then driven locally), a remote handle such as a `RemoteEnv`, or a bare address string the loop dials:

```python
result = model.run("tcp://127.0.0.1:5555", seeds=range(100))
```

The module-level {func}`~rlmesh.run` is the same loop over an explicit `(model, env)` pair. Its `model` argument is a {class}`~rlmesh.Model` subclass class or instance, or a served `RemoteModel` / `SandboxModel` handle. A bare predict function is not accepted as-is: wrap it in the framework `Model` whose arrays it expects, because the library never picks one for you. Pass `rlmesh.RANDOM_SAMPLE` for a baseline that samples the action space and ignores observations:

```python
import rlmesh
import rlmesh.numpy

result = rlmesh.run(rlmesh.numpy.Model(my_policy_fn), env, seeds=range(10))  # arrays in
baseline = rlmesh.run(rlmesh.RANDOM_SAMPLE, env, episodes=10)
```

### Arguments

| Argument            | Default | Meaning                                                                                                            |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `seeds`             | `None`  | One reset seed per episode; alone, it also sets the episode count.                                                 |
| `episodes`          | `None`  | Exact number of episodes to run (default one; must match `len(seeds)` when both are given; `0` runs none).         |
| `execution_horizon` | `1`     | Actions executed per predicted chunk; only engages on a chunk corner (see [below](#execution-horizon-end-to-end)). |
| `hooks`             | `None`  | A {class}`~rlmesh.RunHooks` observing the loop (see [below](#watching-and-capping-the-loop)).                      |
| `instruction`       | `None`  | Text written into a spec'd model's text input every step, in its declared shape.                                   |
| `close_env`         | `False` | Shut the env down when the run finishes (opt-in).                                                                  |
| `trial_index_base`  | `0`     | First trial ordinal; episode `i` walks `trial_index_base + i` (see [trial ordinals](#trial-ordinals)).             |
| `workflow_edition`  | `None`  | Workflow edition this run is evaluated under (see [below](#declare-a-workflow-edition)).                           |

With neither `seeds` nor `episodes`, `run()` does a single episode. `execution_horizon` is accepted by both the bound methods (`model.run` / `model.session`) and the module-level {func}`~rlmesh.run` / {func}`~rlmesh.session`, which forwards it through.

`run()` drives the native runtime loop -- the same engine that drives a served model -- so a vectorized env (`num_envs > 1`) runs through the identical call, with all lanes batched into each predict (the batch corners in {doc}`models`). `hooks=` and `instruction=` work on this loop and on the session loop alike. Only the live viewer is a session option: `view=` is a {func}`~rlmesh.session` parameter, and the module-level `rlmesh.run` forwards it for a served handle or `RANDOM_SAMPLE` (which run on the session loop) but refuses it for a local model.

### Declare a workflow edition

`workflow_edition=` pins the semantics a run or session is evaluated under — the runtime's own declaration, above every other surface. Without it the run takes `RLMESH_WORKFLOW_EDITION`, then the model class's `workflow_edition`, then `[tool.rlmesh] workflow_edition`; with nothing declared anywhere it floats to this build's newest edition and says so once. An edition no participant can run is refused before the first episode, naming what each tier wants and can do. The precedence table and what to paste is in {doc}`../editions/index`.

### Trial ordinals

Every episode walks a trial ordinal: episode `i` is trial `trial_index_base + i`, base `0` unless you pass one. The ordinal reaches the env as `reset(options={"trial_index": ...})` only if the env declared the key in [`EnvFactory.reset_options`](environments/reference.md#reserved-reset-options) -- an env that did not never sees it -- and is recorded on every episode's {attr}`EpisodeResult.trial <rlmesh.EpisodeResult.trial>` either way. A benchmark env that sweeps a fixed list of initial states or goals therefore walks them in order by default, and `trial_index_base` lets a local eval reproduce one shard of a platform run (shard `s` of `M` episodes each is `trial_index_base=s * M`). A non-zero base needs the runtime to own resets, like `seeds`; an autoresetting vector env mints no ordinal.

### Watching and capping the loop

`run()` also takes `max_episode_steps` and `max_episode_seconds`, per-episode caps that mark a capped episode `truncated` exactly like an env time limit (runtime-enforced, so they need the runtime to own resets -- an autoresetting vector env is driven with `episodes` instead).

To _observe_ the loop, pass `hooks=`, a {class}`~rlmesh.RunHooks` subclass whose overrides observe it: `on_run_start` (with a {class}`~rlmesh.RunContext` for role reads and frame discovery), `on_episode_start`, `on_step` (with a {class}`~rlmesh.StepEvent` carrying the observation, action, reward, the step's own `terminated` / `truncated`, per-step timings, and a lazy role `read`), `on_episode_end`, and `on_run_end`. The same callbacks fire in the same per-episode order on `run()` and on {meth}`Session.run <rlmesh.Session.run>`; on a vectorized env, episodes interleave. Every default is a no-op, a hook exception aborts the run with its own type, hooks never change the returned result, and `on_run_end` always fires once with the completed episodes -- enough for progress bars, per-step logging, or streaming metrics without writing the loop yourself:

```python
class Progress(rlmesh.RunHooks):
    def on_episode_end(self, result):
        print(f"episode {result.index}: reward {result.reward:.2f}")

result = model.run(env, seeds=range(50), max_episode_steps=500, hooks=Progress())
```

### The result

{class}`~rlmesh.RunResult` is immutable and aggregates its episodes:

| Member          | Type                        | Meaning                                                                                     |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| `.episodes`     | `tuple[EpisodeResult, ...]` | One {class}`~rlmesh.EpisodeResult` per episode.                                             |
| `.mean_reward`  | `float`                     | Mean total reward across episodes.                                                          |
| `.success_rate` | `float \| None`             | Fraction of episodes the env reported as a success; `None` if any episode lacks the signal. |
| `.num_episodes` | `int`                       | Episode count.                                                                              |
| `.total_steps`  | `int`                       | Summed steps across episodes.                                                               |

Each {class}`~rlmesh.EpisodeResult` carries `index`, `seed`, `steps`, `reward`, `terminated`, `truncated`, and `success`:

```python
for ep in result.episodes:
    print(ep.index, ep.seed, ep.steps, ep.reward, ep.terminated, ep.success)
```

```{caution}
`success_rate` counts only the env's own task outcome: Gymnasium's `info["is_success"]` or `info["success"]`, captured per episode as the `success` field on {class}`~rlmesh.EpisodeResult`. It is `None` when the run is empty or any episode lacks that flag; a terminal state is never read as success. If a terminal state *is* the metric you want, count `terminated` over `result.episodes` yourself, or have the env report success through `info`.
```

`.advisories` holds each distinct `rlmesh.adapters.Advisory` the runtime raised while relaying data between the env and the model. A `"caution"` means the runtime converted data for a peer that could not read it as sent, so the model may not have seen exactly what the env produced. The open-source runtime never converts: it refuses a payload a peer cannot decode, so this tuple is empty.

## `session()`: manual, step-by-step control

`session()` hands back a {class}`~rlmesh.Session` you drive yourself. Use it as a context manager so the env connection (and any managed model) closes on exit:

```python
with model.session(env, instruction="put the cup on the plate") as sess:
    obs, info = sess.reset(seed=0)
    while not sess.done:
        action = sess.predict(obs)
        obs, reward, terminated, truncated, info = sess.step(action)
```

The loop primitives mirror Gymnasium, with the adapter folded in.

```{mermaid}
flowchart LR
    reset["reset(seed)"] --> predict["predict(obs)"]
    predict --> step["step(action)"]
    step -->|not done| predict
    step -->|done| reset
```

- `sess.reset(seed=None, trial_index=None)` → `(obs, info)`. Begins an episode; ends the previous one (firing `on_episode_end`) and clears adapter state such as the frame-stack buffer. `trial_index` is the 0-based ordinal of this episode in a benchmark's trial sweep; it reaches the env as `reset(options={"trial_index": ...})`, but only if the env declared the key in [`EnvFactory.reset_options`](environments/reference.md#reserved-reset-options) -- passing one to an env that did not warns and resets without it. `sess.run(trial_index_base=0)` walks the ordinals for you (episode `i` is trial `trial_index_base + i`), delivers each to a declaring env, and reports each on {attr}`EpisodeResult.trial <rlmesh.EpisodeResult.trial>` whether or not the env asked for it.
- `sess.predict(obs)` → `action`. Applies the model's adapter around the model's own predict: the declarative obs transform, host-side frame stacking, any {class}`~rlmesh.adapters.Custom` code, instruction injection into declared text leaves, and chunk replay (one action per call). Returns an env-ready action.
- `sess.step(action)` → `(obs, reward, terminated, truncated, info)`. Applies the action and records reward and termination.
- `sess.done` is `True` once the current episode terminated or truncated.
- `sess.close()` releases the connection, shuts the env down only on the `close_env` opt-in, and fires `on_close`.

Drive multiple episodes by hand when you want to branch on each step: render a frame, apply your own stop condition, or fork the rollout.

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

`on_episode_end` fires at every episode boundary (the next `reset()`, or `close()` for the last episode), so a stateful model clears its per-episode state identically whether you drive by hand or call `run()`. {meth}`Session.run <rlmesh.Session.run>` pumps whole episodes through these same primitives; `model.run(...)` drives the same episodes on the native runtime loop.

## `read` and `reader`: inspect observations by role

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

- A bare role constant (`adapt.EEF_POS`, `adapt.IMAGE_PRIMARY`), kept in the env's native encoding and using the env's own declared layout.
- A model-input leaf that declares the encoding you want, such as `adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc")` or `adapt.State(adapt.EEF_POS)`. The adapter converts to that form whatever the env stores.

Roles and leaves are the same vocabulary the rest of the adapter system uses; see {doc}`adapters`. The env must publish adapter tags (via an {class}`~rlmesh.EnvFactory` or `rlmesh.adapters.tag(...)`), or the read raises an `AdapterResolutionError`, since there are no roles to address otherwise. Values come back in the env's own framework (NumPy for a Gymnasium env, torch for a torch route).

Reach for it to debug an env: confirm what a camera returns (shape, layout, value range) without threading it through a model. It also gives a consistent way to log canonical roles, recording `EEF_POS` or the primary image the same way across heterogeneous envs, since the role addresses the quantity rather than the env's key. The same read works for reward shaping: compute a shaped term over canonical roles, e.g. `reward - 0.1 * distance(sess.read(obs, adapt.EEF_POS), goal)`.

## Execution horizon, end to end

A policy that emits an action _chunk_ (ACT, diffusion, flow, VLA action heads) defines a `predict_chunk` or `predict_chunk_batch` corner (see the four corners in {doc}`models`). `execution_horizon` tells the rollout how many actions of each predicted chunk to apply before re-planning:

```python
result = model.run(env, seeds=range(50), execution_horizon=8)
```

The runtime owns the replay. It calls the model once, executes the first `execution_horizon` actions of the returned chunk one per env `step`, then calls the model again.

```{mermaid}
flowchart LR
    predict["predict_chunk(obs)"] --> chunk["native chunk [H, ...]"]
    chunk --> prefix["execute first execution_horizon<br/>actions, one per env step"]
    prefix -->|chunk exhausted| predict
```

One action is applied per step regardless. The model returns its whole native chunk and the runtime executes `min(len(chunk), execution_horizon)` of it; an autoregressive head that declares `execution_horizon` can instead decode exactly that many. A chunk corner that slices itself down to the horizon is discarding actions the runtime was about to use, not saving work.

A model whose chunk length K is fixed declares it (`native_chunk = K`, see {doc}`models`). The runtime then refuses an `execution_horizon` above K when the adapter resolves, instead of quietly re-planning every K steps of an H-step plan, and fails a predict whose chunk is not exactly K long. Undeclared, the contract is elastic: a short chunk is replayed as far as it goes and warns once.

`execution_horizon` only matters when the model defines a chunk corner. Requesting `execution_horizon > 1` on a model with no `predict_chunk` warns and runs un-chunked (one fresh prediction per step), so the default of `1` is always safe. The horizon is bounded at 1024, and it cannot be combined with a lockstep vector env (a gym vector env served as one endpoint): chunk replay is whole-batch there, so one lane's episode end would discard every lane's buffered frames. A lane endpoint (`EnvServer([env, ...])`) replays per lane and takes any horizon.

## Where next

- {doc}`models`: the four predict corners, the lifecycle seams, and the batched-observation fusion this loop drives.
- {doc}`adapters`: how environment tags and a model spec resolve into the adapter every entry point applies.
- {doc}`serving-environments`: addresses, readiness, and health for the remote envs `run()` and `session()` dial.
- {doc}`../api/core`: the autodoc signatures for the top-level `run` / `session` entry points and the result types.
