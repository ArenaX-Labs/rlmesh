# Debugging adapters

When a model trained against one environment underperforms against another, the cause is usually a misaligned adapter: an image that came through channels-first, a rotation in the wrong packing, a proprio vector scaled differently than the policy expects. RLMesh gives you three ways to see what the adapter actually does before and during a run, so you can confirm the bridge matches your intent instead of inferring it from a bad success rate.

The three tools, from static to live: `adapter.describe()` prints the exact transforms the resolver chose; the `read` / `reader` API extracts any role from a raw observation in whatever encoding you ask for; and join advisories warn you, at authoring time, about a tag that looks mis-declared. Start with `describe()`, reach for `read` when you need to see the values, and lean on advisories to catch mistakes before a peer ever connects.

## Read what the resolver chose

{func}`~rlmesh.adapters.resolve` returns an {class}`~rlmesh.adapters.Adapter`, and `adapter.describe()` prints every transform it derived: each resize, layout transpose, encoding conversion, range map, key remap, slice, and clip. Call it once, before you run a step.

```python
import rlmesh.adapters as adapt

adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
print(adapter.describe())
```

For the manipulation pair in {doc}`adapters`, the output shows the primary image resized, the rotation going `quat_xyzw -> rot6d`, the instruction key remapped (`goal -> task`), and the model's 6-D action rotation converted back to the env's 3-D `axis_angle` and clipped. If a transform you expected is missing, or one you did not intend is present, the spec and the tags disagree about that leaf. When the spec uses a {class}`~rlmesh.adapters.CustomEncoding`, `describe()` also lists the host-side repack arms under a `host-side encodings:` section.

On the served path, the same description is available from the resolved adapter the model server builds, and from the contract a client receives. The point is the same either way: read the description, not the success rate.

## Inspect observations by role

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

## Join advisories and warnings

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

## Symptom, cause, fix

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

- {doc}`adapters` and {doc}`adapters/reference` — the declarative pipeline, the full field reference, and the conversion policy these tools surface.
- {doc}`adapters/escape-hatches` — when a misalignment needs host code: {class}`~rlmesh.adapters.Custom`, {class}`~rlmesh.adapters.CustomEncoding`, or an {class}`~rlmesh.adapters.AdapterBase` subclass.
- {doc}`evaluation` — the `read` / `reader` API in the context of the full eval loop.
- {doc}`error-handling` — when a mismatch is fatal: the exceptions resolution raises and how to recover.
- {doc}`serving-environments` and {doc}`remote-clients` — confirming an env publishes the tags a remote model resolves against.
