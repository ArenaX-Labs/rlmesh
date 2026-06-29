# Troubleshooting

Most of what can go wrong in an eval surfaces at one of two seams: adapter resolution (a model spec that does not line up with an env's tags and spaces) or the transport (a remote env or served model that is unreachable, slow, or fails mid-episode). RLMesh reports each as a distinct exception so you can tell a wiring mistake from a runtime fault and recover the right way.

The first half of this page maps the exceptions {func}`~rlmesh.adapters.resolve`, `serve`, `predict`, and the eval loop ({func}`~rlmesh.run` / {func}`~rlmesh.session`) raise, what causes each, and how a run behaves when a connection drops or a model crashes part-way through an episode. The second half covers the inspection tools for a misaligned adapter that resolves cleanly but behaves wrong: `describe()`, the `read` / `reader` path, the live viewer, and join advisories.

## The exception families

Two families cover almost everything.

Resolution failures raise {exc}`~rlmesh.adapters.AdapterResolutionError`, a subclass of `ValueError`. It comes from {func}`~rlmesh.adapters.resolve`, {func}`~rlmesh.adapters.resolve_from_contract`, {func}`~rlmesh.adapters.tag`, the read API, and the adapter resolution that `run`/`session` do on connect. It always points at the offending leaf and what it expected, and it fires before a single step runs.

Runtime failures come from the native core and reach Python through a small exception hierarchy plus a few standard built-ins. The native module defines `RLMeshException` (a subclass of `RuntimeError`) as the base, with `ProtocolException` and `EnvironmentException` beneath it. An environment that reports a fault while serving a request raises `EnvironmentException`. Transport faults, timeouts, and bad arguments map to the standard `ConnectionError`, `TimeoutError`, and `ValueError` instead, so ordinary `except` clauses catch them without importing anything RLMesh-specific.

```{note}
`RLMeshException`, `ProtocolException`, and `EnvironmentException` live in the
native module (`rlmesh._rlmesh`). `EnvironmentException` is the one the env path
raises today; `ProtocolException` is reserved for protocol-level faults, and the
current boundary surfaces generation/handshake mismatches as `RuntimeError`
rather than that type. Catch `RLMeshException` to cover the whole family at once.
```

## What each call raises

### `resolve()` and `resolve_from_contract()`

Both raise {exc}`~rlmesh.adapters.AdapterResolutionError` when the model spec cannot be bridged to the env's tags and spaces: a required role with no `optional` fallback, a declared channel mismatch, an upscale without `allow_upscale`, an aspect mismatch without `fit`, an unsupported `resample` or `dtype`, an impossible encoding conversion, an unknown field on a known leaf, or a join-time class/width/encoding/range disagreement. {func}`~rlmesh.adapters.resolve_from_contract` adds two of its own: the contract carries no adapter tags, or those tags are not serializable JSON.

```python
import rlmesh.adapters as adapt

try:
    adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
except adapt.AdapterResolutionError as exc:
    print(exc)  # names the leaf and what it expected
```

A spec that references a `Custom` input by `entrypoint=` also raises here unless you pass `resolve(..., trust_entrypoints=True)`. The conversion-policy table in {doc}`adapters/reference` decides which conversions apply silently, warn, or fail.

### `run()`, `session()`, and `Session.predict()`

The eval loop resolves the adapter on the first connect, so a spec/env mismatch raises {exc}`~rlmesh.adapters.AdapterResolutionError` before any episode begins. A few more checks run at the same point:

- Pointing {func}`~rlmesh.run` at a vector endpoint (`num_envs > 1`) raises `ValueError`. The per-episode loop reads scalar reward and termination, so a vector env is rejected up front rather than crashing deep in the step loop. Serve it as a single env, or drive it with `RemoteVectorEnv` directly.
- An env that publishes adapter tags paired with a model whose `spec` is `None` raises {exc}`~rlmesh.adapters.AdapterResolutionError`: pass a `ModelSpec`, or `rlmesh.NO_ADAPTER` if the model adapts its own observations.
- A target that is neither an env, an `EnvFactory`, a remote handle, nor an address string raises `TypeError`.

Once running, `predict` and `step` surface whatever the env or model raises. For a local model, `predict` runs in-process, so a bug in your prediction function propagates as its own native Python exception. For a served model, the server maps a handler that declines a request to `RuntimeError` (`"model error: ..."`).

### `serve()`

{class}`~rlmesh.EnvServer` validates published tags against the env's spaces at construction (for a scalar env), so a bad tag raises {exc}`~rlmesh.adapters.AdapterResolutionError` at startup instead of when the first model connects. Asking for `device=` on a numpy env, or combining an explicit `address` with `host`/`port`/`path`, raises `ValueError`. A bind that fails surfaces as `RuntimeError` (`"server error: ..."`).

A served model resolves its adapter once per env, at the configure step rather than at connect, so a spec/env mismatch fails route configuration loudly with {exc}`~rlmesh.adapters.AdapterResolutionError` rather than predicting wrongly.

## Symptom, cause, fix: resolution and transport

| Symptom                                                        | Cause                                                           | Fix                                                                          |
| -------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `AdapterResolutionError: ... no usable counterpart`            | a model input or actuator role the env never tags               | tag the role on the env, mark the input `optional`, or drop it from the spec |
| `AdapterResolutionError: env contract carries no adapter tags` | resolving from a contract on an untagged env                    | serve with `tags=` or {func}`~rlmesh.adapters.tag`; see {doc}`adapters`      |
| `AdapterResolutionError: ... has spec=None`                    | a tagged env paired with a model that declares no spec          | pass `spec=<ModelSpec>`, or `rlmesh.NO_ADAPTER` to opt out of adaptation     |
| `ValueError: ... reports num_envs=...`                         | {func}`~rlmesh.run` aimed at a vector endpoint                  | serve a single env, or use `RemoteVectorEnv` for the vector endpoint         |
| `ValueError: Endpoint ... serves N environments`               | `RemoteEnv` connected to a multi-env endpoint                   | connect with `RemoteVectorEnv` instead (see {doc}`remote-clients`)           |
| `ConnectionError`                                              | the env/model endpoint is unreachable or the connection dropped | confirm the address and that the server is up; re-dial a fresh client        |
| `TimeoutError`                                                 | a connect or request exceeded its deadline                      | check the endpoint is serving; retry the operation                           |
| `EnvironmentException`                                         | the env reported NotReady, Busy, Internal, Crashed, or Closed   | reset the env, or inspect the env-side logs for the underlying fault         |
| `RuntimeError: model error: ...`                               | a served model handler declined the request                     | check the model's prediction code against the obs it actually receives       |
| `ValueError: device=... requires a torch/jax ...`              | `device=` on a numpy env/model                                  | drop `device=`, or set `framework="torch"` / `"jax"`                         |
| `ImportError: Failed to import _rlmesh native module`          | the compiled extension is missing                               | reinstall the wheel for your platform                                        |

## Connection loss and crashes mid-episode

The eval loop does not retry a failed step. When the env connection drops or a served model fails while an episode is in flight, the call (`predict` or `step`) raises, and {func}`~rlmesh.run` propagates it. Before the exception leaves `run`, the loop still runs cleanup: it ends the open episode (firing a stateful model's `on_episode_end`), calls `on_close`, and closes the session. The in-progress episode is discarded, and `run` raises rather than returning a partial {class}`~rlmesh.RunResult`.

```{mermaid}
flowchart TD
    A[run starts an episode] --> B[predict, then step]
    B -->|ok, not done| B
    B -->|env or model fault| E[predict/step raises]
    B -->|terminated or truncated| C[record EpisodeResult]
    C -->|more episodes| A
    C -->|done| R[return RunResult]
    E --> F[end open episode: on_episode_end, on_close, close]
    F --> G[re-raise the exception]
```

A few consequences follow from this.

A transport fault during an established run is reported as `ConnectionError`, not as a recoverable retry. RLMesh classifies transport conditions internally (a server that is still binding is treated as transient), but the public Python clients do not reconnect a dropped session for you. To recover, construct a fresh client and start a new run.

A served model handler that raises becomes a `RuntimeError` carrying the handler's message. The env stays up, so you can fix the model and re-dial without restarting the env server.

```{caution}
Per-step request timeouts and connect timeouts exist on the native client but are
not exposed through `RemoteEnv` / `RemoteVectorEnv` or the `run`/`session` loop
today. A step against an env that hangs will block. Bound it at the env: serve
with `ServeOptions(idle_timeout_seconds=...)` so an idle server stops on its own,
and supervise the process. See {doc}`performance` for the session lifecycle.
```

## Recovering cleanly

Wrap a run in a normal `try`/`except` and decide per family. Resolution errors are wiring bugs you fix in the spec or tags; transport errors call for a re-dial; an `EnvironmentException` usually means resetting or restarting the env.

```python
import rlmesh
import rlmesh.adapters as adapt

try:
    result = model.run(env, seeds=range(100))
except adapt.AdapterResolutionError as exc:
    raise SystemExit(f"adapter mismatch, fix the spec or tags: {exc}")
except (ConnectionError, TimeoutError) as exc:
    ...  # re-dial a fresh client and retry the run
except rlmesh._rlmesh.EnvironmentException as exc:
    ...  # the env reported a fault; reset or restart it
```

Using `session()` as a context manager (or `run()`, which closes for you) guarantees the connection and any managed model are released even when an episode raises, so the next attempt starts clean.

## Debugging a misaligned adapter

When a model trained against one environment underperforms against another, the cause is usually a misaligned adapter: an image that came through channels-first, a rotation in the wrong packing, a proprio vector scaled differently than the policy expects. This resolves cleanly, so it shows up as a bad success rate rather than an exception. RLMesh gives you three ways to see what the adapter actually does before and during a run.

The three tools run from static to live: `adapter.describe()` prints the exact transforms the resolver chose; the `read` / `reader` API extracts any role from a raw observation in whatever encoding you ask for; and join advisories warn you, at authoring time, about a tag that looks mis-declared. Start with `describe()`, reach for `read` when you need to see the values, and lean on advisories to catch mistakes before a peer ever connects.

### Read what the resolver chose

{func}`~rlmesh.adapters.resolve` returns an {class}`~rlmesh.adapters.Adapter`, and `adapter.describe()` prints every transform it derived: each resize, layout transpose, encoding conversion, range map, key remap, slice, and clip. Call it once, before you run a step.

```python
import rlmesh.adapters as adapt

adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
print(adapter.describe())
```

For the manipulation pair in {doc}`adapters`, the output shows the primary image resized, the rotation going `quat_xyzw -> rot6d`, the instruction key remapped (`goal -> task`), and the model's 6-D action rotation converted back to the env's 3-D `axis_angle` and clipped. If a transform you expected is missing, or one you did not intend is present, the spec and the tags disagree about that leaf. When the spec uses a {class}`~rlmesh.adapters.CustomEncoding`, `describe()` also lists the host-side repack arms under a `host-side encodings:` section.

On the served path, the same description is available from the resolved adapter the model server builds, and from the contract a client receives. The point is the same either way: read the description, not the success rate.

### Inspect observations by role

`describe()` tells you the plan. The `read` / `reader` API tells you the values. Both give a read-only, role-addressed view of a raw observation through the same adapter pipeline a model uses, so they are encoding-agnostic across envs and never mutate the observation.

`sess.reader(*items)` resolves once and returns a callable mapping a raw observation to a `{role: value}` dict:

```python
import rlmesh.adapters as adapt

with model.session(env) as sess:
    read = sess.reader(adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc"), adapt.EEF_POS)
    obs, _ = sess.reset(seed=0)
    view = read(obs)
    print(view[adapt.IMAGE_PRIMARY].shape, view[adapt.IMAGE_PRIMARY].dtype)
    print(view[adapt.EEF_POS])
```

`sess.read(obs, item)` is the one-shot single-role form; the underlying reader is cached per item, so calling it every step does not re-resolve:

```python
ee = sess.read(obs, adapt.EEF_POS)
img = sess.read(obs, adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc"))
```

An item is one of two things, and the difference is exactly what you want when debugging:

- A bare role constant (`adapt.EEF_POS`, `adapt.IMAGE_PRIMARY`) returns the quantity in the env's own declared encoding and layout. Use it to confirm what the camera actually sends, before any conversion.
- A model-input leaf that declares an encoding (`adapt.Image(adapt.IMAGE_PRIMARY, layout="hwc")`, `adapt.State(adapt.EEF_ROT, encoding="rot6d")`) returns the value converted to that form. Use it to confirm the conversion produces what the model expects.

Reading the same role both ways, bare and as a leaf, shows you the transform end to end. The env must publish adapter tags (via an {class}`~rlmesh.EnvFactory` or {func}`~rlmesh.adapters.tag`); otherwise there are no roles to address and the read raises {exc}`~rlmesh.adapters.AdapterResolutionError`. Values come back in the env's own framework, NumPy for a Gymnasium env, torch for a torch route. The reader vocabulary and the read API are covered alongside the eval loop in {doc}`evaluation`.

```{tip}
A bare role read keeps the env's native layout, so a `chw` camera comes back
`chw`, not silently transposed. To see the model's view, pass an explicit
`Image(role, layout=...)` leaf with the layout the model declares.
```

### See it live

For a moving target, attach the built-in viewer with `view=` on `run` or `session`. It shows the env's `render()` frame plus every declared camera role, selectable at runtime, with a step/reward HUD.

```python
model.run(env, seeds=range(5), view="terminal")   # half-block frames in the terminal
model.run(env, seeds=range(5), view="http:9000")   # serve frames over HTTP on :9000
```

The string shorthands are `"terminal"`, `"http"`, `"http:PORT"`, and `"both"`; construct `rlmesh.View(...)` directly to tune `fps`, image `format`, or `quality`. The viewer is best-effort: any setup failure disables it with a warning and never breaks the eval. It draws the same image roles `read` exposes, so what you see is what the adapter feeds the model.

### Join advisories and warnings

Some mismatches are not resolution errors, they are smells: a layout that looks mis-declared, a camera the env never provides that a model wants zero-filled, an aspect crop that discards pixels. RLMesh surfaces these as non-fatal advisories at two points.

At authoring time, {func}`~rlmesh.adapters.tag` runs the native join check against the env's spaces and emits each advisory through Python's `warnings`, so you see your own mistake when you tag the env rather than in a peer's serve logs.

```python
import warnings
import rlmesh.adapters as adapt

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    env = adapt.tag(env, tags)
for w in caught:
    print(w.message)   # e.g. a layout hint for a frame that looks CHW
```

At resolve time, an adapter that fabricates or drops data records it. `adapter.advisories()` returns that subset of the description, the per-env data-loss and fabrication notes (a zero-filled camera, an aspect crop), empty when nothing noteworthy happened. A managed runner can log these without failing the run.

```python
for note in adapter.advisories():
    print(note)
```

```{note}
Advisories are deliberately non-fatal. A zero-filled optional camera and a chosen
`fit="crop"` are legitimate, but they change what the model sees, so RLMesh tells
you rather than failing. The conversion-policy table in {doc}`adapters/reference`
lists exactly which conversions warn versus error.
```

## Symptom, cause, fix: a run that resolved but behaves wrong

These mirror the resolver pitfalls in {doc}`adapters/reference`, framed for debugging a run that resolved cleanly but behaves wrong.

| Symptom                                     | Cause                                          | Confirm it                                                         | Fix                                                    |
| ------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------ |
| Image looks scrambled or rotated 90°        | HWC vs CHW layout mismatch                     | `read` the role bare, then as `Image(role, layout="hwc")`, compare | set `layout` on the model `Image` to what it wants     |
| Wrong channel count slips through           | RGB vs grayscale never declared                | check `read(obs, Image(role)).shape[-1]`                           | set `channels` on the `Image` to make a mismatch error |
| Policy acts as if the arm is mis-oriented   | `quat_xyzw` vs `quat_wxyz`, or wrong base      | `read` the rotation role bare to see the env's packing             | match the env's exact `encoding` on the model side     |
| Proprio values out of the expected range    | scale mismatch between env and model           | `read` the role as a `State` leaf and inspect the magnitudes       | set `range` on the model side to map it                |
| A camera frame is all black                 | env lacks the role, filled by `optional`       | `adapter.advisories()` lists the zero-filled camera                | provide the camera, or accept the fill deliberately    |
| Image edges or aspect look cropped          | `fit="crop"` chose to discard pixels           | `adapter.advisories()` lists the crop                              | use `fit="pad"`, or match the target aspect            |
| `describe()` omits a transform you expected | the spec and tags disagree on that leaf's role | read `describe()` line by line against your spec                   | align the role/encoding on whichever side is wrong     |
| Read raises `AdapterResolutionError`        | the env publishes no adapter tags              | check `env.metadata` for the env-tags key                          | serve with `tags=` or {func}`~rlmesh.adapters.tag`     |

## Where next

- {doc}`adapters` and {doc}`adapters/reference` (the resolution rules and the conversion policy behind every {exc}`~rlmesh.adapters.AdapterResolutionError`, plus the full field reference these tools surface).
- {doc}`adapters/escape-hatches` (when a misalignment needs host code: {class}`~rlmesh.adapters.Custom`, {class}`~rlmesh.adapters.CustomEncoding`, or an {class}`~rlmesh.adapters.AdapterBase` subclass).
- {doc}`evaluation` (how {func}`~rlmesh.run` and {func}`~rlmesh.session` drive episodes, what a {class}`~rlmesh.RunResult` reports, and the `read` / `reader` API in context).
- {doc}`serving-environments` and {doc}`remote-clients` (readiness, health, and the client side of the transport).
