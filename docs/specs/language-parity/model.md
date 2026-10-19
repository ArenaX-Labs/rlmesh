# Model authoring

Part of the [language parity spec](../language-parity.md). Status: Draft —
reviewed round 6. Draft: C# column pending review. Rules, decisions and paths (`py/`, `pyrs/`, `rs/`, `capi/`,
`h`, `hpp`) are in [rules](rules.md).

This area owns `Model`, `Policy`, `PredictContext`, `BatchPredictContext`,
`SpecArg`, `NO_ADAPTER` and `predict_seed` (R-27). The native `Model` is
`rlmesh.Model`, the dependency-free model over raw `rlmesh.types.Value` trees.
`rlmesh.numpy|torch|jax.Model`, de-chunking and `device` exist only for their
array bridges (R-24), so the fallback rules here are the raw-`Value` rules of
`py/_models/base.py:367-403` (`can_dechunk` false).

## Shape

A stateful chunked policy that reads the context, loads weights, and is served.

```python
class Planner(rlmesh.Model):
    spec = model_spec
    native_chunk = 8

    def load(self, checkpoint="default.pt"):
        self.weights = read(checkpoint)

    def predict(self, observation):
        return act(self.weights, observation)

    def predict_chunk(self, observation, execution_horizon, context):
        context["state"]["plans"] = context["state"].get("plans", 0) + 1
        return plan(self.weights, observation, seed=context["predict_seed"])

model = Planner.from_config(checkpoint="ckpt.pt")
model.serve("tcp://0.0.0.0:50051")
greedy = rlmesh.Model(lambda obs: 0)
```

```rust
#[derive(Default)]
struct Planner { weights: Weights }

impl rlmesh::Policy for Planner {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK);
    const CONTEXT: Corners = Corners::PREDICT_CHUNK;
    const NATIVE_CHUNK: Option<u32> = Some(8);
    const LOAD_KEYS: Option<&'static [LoadKey]> = Some(&[LoadKey::optional("checkpoint")]);

    fn spec() -> Option<SpecArg> { Some(model_spec().into()) }

    fn load(&mut self, config: &Kwargs) -> Result<()> {
        self.weights = read(config.get("checkpoint"))?; // read(None) = "default.pt"
        Ok(())
    }

    fn predict(&mut self, observation: Value) -> Result<Value> {
        act(&self.weights, observation)
    }

    fn predict_chunk_with_context(
        &mut self, observation: Value, _execution_horizon: u32, context: &mut PredictContext<'_>,
    ) -> Result<Value> {
        *context.state.get_or_insert_with("plans", || 0u64)? += 1;
        plan(&self.weights, observation, context.predict_seed)
    }
}

let config = Kwargs::from_iter([("checkpoint".to_string(), Value::Text("ckpt.pt".into()))]);
let model = Model::from_config::<Planner>(config)?;
model.serve("tcp://0.0.0.0:50051", ModelServeOptions::default())?;
let greedy = Model::new(|_obs: Value| Ok(Value::Int(0)), ModelOptions::default())?;
```

```cpp
struct Planner {
  static rlmesh::types::SpecArg spec() { return model_spec(); }
  static constexpr std::optional<uint32_t> native_chunk = 8;
  static constexpr std::array load_keys{rlmesh::types::LoadKey{"checkpoint"}};
  Weights weights;

  void load(const rlmesh::Kwargs& config) {
    auto it = config.find("checkpoint");
    weights = read(it == config.end() ? nullptr : &it->second);  // read(nullptr) = "default.pt"
  }
  rlmesh::Value predict(const rlmesh::Value& observation) { return act(weights, observation); }
  rlmesh::Value predict_chunk(const rlmesh::Value& observation, uint32_t execution_horizon,
                              const rlmesh::types::PredictContext& context) {
    ++context.state.get_or_emplace<uint64_t>("plans");
    return plan(weights, observation, context.predict_seed);
  }
};

rlmesh::Kwargs config;
config.insert_or_assign("checkpoint", rlmesh::Value(std::string("ckpt.pt")));
auto model = rlmesh::Model::from_config<Planner>(config);
model.serve("tcp://0.0.0.0:50051");
auto greedy = rlmesh::Model([](const rlmesh::Value&) { return rlmesh::Value(int64_t{0}); });
```

```c
static const RLMeshLoadKey planner_keys[] = {{"checkpoint", false}};
RLMeshModelVtable vt = RLMESH_MODEL_VTABLE_INIT;
vt.load = planner_load;
vt.load_keys = planner_keys;
vt.load_keys_len = 1;
vt.predict = planner_predict;
vt.predict_chunk = planner_predict_chunk;
vt.context_corners = RLMESH_CORNER_PREDICT_CHUNK;
vt.has_native_chunk = true;
vt.native_chunk = 8;
vt.spec = (RLMeshSpecArg){RLMESH_SPEC_KIND_MODEL_SPEC, spec};
vt.free_user_data = planner_free;

RLMeshModel* model = NULL;
rlmesh_model_from_config(&vt, planner_alloc(), config /* RLMeshValue* DICT */, &model);
rlmesh_model_serve(model, "tcp://0.0.0.0:50051", NULL);
rlmesh_model_free(model);
```

```csharp
sealed class Planner : IPredict, IPredictChunkWithContext, ILoad, IModelDeclarations
{
    static readonly ModelDeclarations Declared = new()
    {
        Spec = ModelSpecs.Planner(),
        NativeChunk = 8,
        LoadKeys = new[] { new LoadKey("checkpoint") },
    };
    public ModelDeclarations Declarations => Declared;

    Weights weights;

    public void Load(Metadata config) =>
        weights = Read(config.TryGetValue("checkpoint", out var c) ? c : null); // Read(null) = "default.pt"

    public Value Predict(Value observation) => Act(weights, observation);

    public Value PredictChunk(Value observation, uint executionHorizon, PredictContext context)
    {
        context.State["plans"] = (context.State.TryGetValue<long>("plans", out var n) ? n : 0) + 1;
        return Plan(weights, observation, context.PredictSeed);
    }
}

var config = new Metadata { ["checkpoint"] = "ckpt.pt" };
using var model = Model.FromConfig(new Planner(), config);
model.Serve("tcp://0.0.0.0:50051");
using var greedy = new Model((Value obs) => 0);
```

## Parity table

Semantics first (identical in every language), then the symbol tables.

### Authoring

- Two pieces in every language: a policy that supplies the members a Python
  `Model` subclass overrides (corners, `load` and its keys, `on_episode_end`, `close`,
  `spec`, `native_chunk`, `allow_fusion`, `workflow_edition`), and the concrete
  `Model` that owns the episode store, the resolved spec and the hooks. Rust: a
  `Policy` trait plus `Model`. C: `RLMeshModelVtable` plus `RLMeshModel*`.
  C++: any class or callable, detected by its members, plus `rlmesh::Model`.
  C#: a class implementing `IPolicy`'s corner and hook interfaces, or a
  delegate, plus `RLMesh.Model` (MDL-23).
- A native policy type is the counterpart of a Python `Model` subclass, for
  corners, hooks, `load` and declarations alike: its `on_episode_end` is the
  episode-end hook, its `load` runs once in the constructor, and its
  declarations are the model's (`py/_models/base.py:597-618`). A native
  closure is the counterpart of a wrapped bare callable. Python's wrapping
  `Model(obj)` of a duck-typed policy object keeps two Python-only
  conveniences for third-party objects, not carried over (MDL-10): its
  episode-end hook is the object's `reset` (`getattr(inst, "reset", None)`;
  the object's `on_episode_end` is not read, `py/_models/_coerce.py:66-77`),
  and it never calls the object's `load()`, because the object's own
  `__init__` is authoritative (`py/_bootstrap/loaders.py:355-381`). Only its
  declarations follow the native rule (MDL-P8).
- Corner and context detection matches Python's override and signature
  detection, per corner (`py/_models/base.py:66-82,119-139,604-610`):
  - Rust has no overloading, so each corner has a context-free method named as
    in Python and a `<corner>_with_context` twin. `const CORNERS: Corners`
    (default `Corners::PREDICT`) lists the corners the impl writes and
    `const CONTEXT: Corners` (default `Corners::NONE`) those that read the
    context, which are called through their `_with_context` method. A plain
    impl writes both consts by hand; there is no proc macro (MDL-3).
  - C: a NULL corner slot is "not defined"; `uint32_t context_corners`
    (`RLMESH_CORNER_*` bits, INIT 0) lists the corners that receive a context.
    Every other corner receives `context == NULL`. A `context_corners` bit
    for a NULL slot, or any bit outside the four `RLMESH_CORNER_*` values, is
    an invalid declaration (below), as Rust's `CONTEXT ⊄ CORNERS` is.
  - C++: members detected at compile time per overload (with or without
    `execution_horizon`, with or without `const PredictContext&`). Hooks keep
    Python's optional parameters too (`py/_models/base.py:87-101`): both
    `on_episode_end()` and `on_episode_end(std::string_view)` are detected,
    and both `load()` and `load(const Kwargs&)`. The one-argument form wins
    when it is callable (as Python passes the id whenever the hook can take
    it); otherwise the nullary form is called. A nullary-only `load()` is
    `load(self)`: it declares `load_keys = {}`, so a nonempty config is
    refused with the unknown-key text (Load), and it may not be combined
    with a `load_keys` member. A single `load` or `on_episode_end` member
    (`&P::load` / `&P::on_episode_end` well-formed) callable in neither form,
    and a nullary-only `load()` beside `load_keys`, are refused by
    `static_assert`, so no such member is silently never called.
  - C# (MDL-23): one interface per corner and form, all extending the
    marker `IPolicy`: `IPredict` / `IPredictWithContext`, `IPredictBatch` /
    `IPredictBatchWithContext`, `IPredictChunk` / `IPredictChunkWithContext`,
    `IPredictChunkBatch` / `IPredictChunkBatchWithContext`. The constructor
    reads them once with `is` and fills C's slots and `context_corners`, so
    "written" means "implemented", as for C's non-NULL slots. A type
    implementing both forms of one corner is refused at construction with
    the TypeError class: Python has one method per corner and C one slot,
    and picking one silently would leave the other never called (the reason
    C++ `static_assert`s on uncallable members). Hooks take one form each:
    `IOnEpisodeEnd.OnEpisodeEnd(string episodeId)` (a nullary author ignores
    the id, as in Rust) and `ILoad.Load(Metadata config)` (Python's nullary
    `load(self)` is `ILoad` with `LoadKeys = []`, as C++'s nullary `load()`
    declares `load_keys = {}`).
- `execution_horizon` is always passed to Rust, C and C# chunk corners; an author
  that ignores it leaves it unused (Python's optional parameter,
  `py/_models/base.py:660-665`). C++ detects whether the member takes it.
- Context opt-in keeps Python's "pays nothing" rule
  (`py/_models/base.py:119-139`): only a corner that reads the context creates
  an episode-store entry or advances `predict_index`. A context-free corner
  never touches the store and never hits the capacity refusal. A batched
  corner gets one context per row.
- An invalid declaration is refused at construction with Python's `TypeError`
  class (R-21: `Error::Type`, `RLMESH_ERR_TYPE`, `rlmesh::TypeError`): no
  derivable `predict` (`base.py:705-709`), a static `native_chunk` with no
  chunk corner (`base.py:728-740`), a loaded `native_chunk` (Rust
  `fn native_chunk(&self)`, C `loaded_native_chunk`, C++
  `loaded_native_chunk()`) with no chunk corner, checked after `load` (Load;
  MDL-P5), a `CONTEXT` corner missing from `CORNERS`
  (C: a `context_corners` bit for a NULL slot or outside the four
  `RLMESH_CORNER_*` values, `RLMESH_ERR_TYPE`).
  C++ also `static_assert`s on what it detects statically. C, C++ and C#
  detect what is written (non-NULL slots, detected members, implemented
  interfaces), so they refuse every such case. Rust cannot see which trait methods an impl wrote: it checks the
  consts, not the bodies, so the consts can disagree with the bodies in both
  directions, where Python detects every written corner by override
  (`py/_models/base.py:599-603`). An impl that lists a corner in `CORNERS`
  (or `CONTEXT`) without writing that method, including an empty
  `impl Policy for T {}` under the `Corners::PREDICT` default, constructs, and
  the first call to that corner fails with the builtin `RuntimeError` class
  naming the entry (Corners), where Python refuses at construction with
  `TypeError`. A corner method that is written but not listed in `CORNERS`
  is never called: the model behaves as if it were absent (a `predict_batch`
  left out runs per lane; a `predict_chunk` left out runs un-chunked, with
  only R-23's horizon `UserWarning`), where Python would call it. A
  `_with_context` twin whose corner is not listed in `CONTEXT` is never
  called either; the context-free method is called instead. Both are inherent to
  const declarations without a proc macro (MDL-3).

### Fallback rules

Normative in every language; the table illustrates them.

- Debatching is always allowed. De-chunking is never allowed for raw `Value`
  (MDL-6). Derivation never adds an axis: a missing batched corner is covered
  by the engine's per-lane loop, a missing chunk corner means the model runs
  un-chunked.
- When `predict` could come from either corner, it comes from `predict_batch`.
- `native_chunk = K` is a promise: it needs a chunk corner (TypeError class,
  for a static K and a loaded K alike, both refused at construction; Load),
  must be `>= 1`, refuses an `execution_horizon > K` at resolve, and fails any
  chunk that is not exactly K frames. A K below 1 (static or loaded, including
  C `has_native_chunk` with `native_chunk = 0`), a horizon above K, a chunk of
  the wrong length and an empty chunk at a horizon above 1 all fail with the
  ValueError class (`Error::InvalidArgument`, `RLMESH_ERR_INVALID_ARGUMENT`,
  `rlmesh::ValueError`), as Python's local Session raises `ValueError` for each
  (`py/_models/_chunk.py:123-129,144-147`, `py/_models/_eval.py:775-788`;
  the bridges' de-chunk path checks K the same way, `py/_models/base.py:252-288`,
  but it never runs for raw `Value`, MDL-6). That is the class wherever the
  chunk is replayed in the process running the model: a local Session,
  `Model.run`'s in-process engine and a served model's engine alike. Today
  the native engine raises `Error::Model` for the wrong-length and empty
  chunk (`rs/model/engine.rs:614-622,727`), so Python's `Model.run` raises
  `RuntimeError("model error: …")` (`pyrs/types.rs:50`); both align (Rust
  current differences, MDL-P10). Across the wire the runtime sees R-21's
  Model class (R-12).
- `native_chunk = None` (undeclared) is elastic: the runtime replays the
  prefix `min(len(chunk), execution_horizon)` (`py/_models/base.py:509-518`;
  an empty chunk at a horizon above 1 is the ValueError class, above). A chunk
  shorter than the horizon warns `RuntimeWarning` (R-23) with Python's text
  (`py/_models/_chunk.py:136-141`), at most once per episode: Python's Session
  warns once per `ChunkReplay`, which it rebuilds at connect and at every
  episode boundary (`py/_models/_eval.py:820,930,1067`). Natively the engine
  keys that once-flag by the lane's episode (`episode_id`; once per route for
  an anonymous lane). Today it warns once per endpoint through `tracing::warn!`
  (`rs/model/engine.rs:600-640`), so Python's `Model.run` emits no Python
  warning (Rust current differences, MDL-P11). R-23's default handler then
  prints each distinct message once per process.
- `allow_fusion` (default `true`) lets a grouped predict put lanes from
  different envs into one batched-corner call.

| Authored corners                                      | `predict`                 | `predict_batch` | `predict_chunk`                       | `predict_chunk_batch` |
| ----------------------------------------------------- | ------------------------- | --------------- | ------------------------------------- | --------------------- |
| `predict`                                             | own                       | none            | none                                  | none                  |
| `predict_batch`                                       | batch of one, row 0       | own             | none                                  | none                  |
| `predict` + `predict_batch`                           | own                       | own             | none                                  | none                  |
| `predict` + `predict_chunk`                           | own                       | none            | own                                   | none                  |
| `predict` or `predict_batch`, + `predict_chunk_batch` | as above                  | as above        | batch of one of `predict_chunk_batch` | own                   |
| chunk corner(s) only                                  | refused (TypeError class) | —               | —                                     | —                     |

Every row whose `predict_chunk` is none (`predict`, `predict_batch`, or
both) runs un-chunked, and `run` / `session` with `execution_horizon > 1`
then warn `UserWarning` (R-23), as Python warns whenever the derived
`predict_chunk` is missing (`py/_models/base.py:1187-1192,1447-1452`).

### Dispatch by route kind

Python's behaviour, the target everywhere.

- **Spec'd route.** Per lane. Batched corners on a vectorized or grouped
  route, chunk corners when `execution_horizon > 1`. Each lane has its own
  context.
- **Spec-less route** (`spec` resolves to none, or `NO_ADAPTER`): one call per
  route request over its N lanes (`rs/model/engine.rs:1133`,
  `pyrs/model.rs:203-304`):
  - N = 1: `predict`, or `predict_chunk` when `execution_horizon > 1` and the
    policy has one, with the lane's context.
  - N > 1 with `predict_batch` and an observation space: `predict_batch` over
    the N per-lane values, per-lane contexts.
  - N > 1 otherwise: `predict` receives one batched value (every leaf stacked
    on a leading axis N) with an anonymous context and returns an action
    batched the same way.
  - Chunk corners are not used when N > 1; `predict_chunk_batch` is never used
    on a spec-less route.
- **A chunk** is a `Value` whose leading axis is frames: a `List` of per-step
  values, or a `Tensor` (or `Dict` of tensors) with leading axis K. A 0-d
  tensor or any other leaf, a `Tuple` included, is one frame
  (`crates/rlmesh-adapters/src/stateful.rs:379-423`). This is the native
  `split_chunk` rule; Python's local Session splits any iterable today, a
  tuple included, and aligns (MDL-P9).

### Episode store

One per `Model`, in the Rust facade, used by every binding.

- Entries are keyed by `episode_id` and tagged with the `env_id` of the route
  that created them. The store never evicts.
- A new episode past capacity is refused with the builtin `RuntimeError` class
  (R-21 "local refusals": `Error::Runtime`, `RLMESH_ERR_RUNTIME`,
  `rlmesh::RuntimeError`); live episodes are untouched. Capacity is 65,536 or
  `RLMESH_EPISODE_CAPACITY`, read once at construction; a non-digit value is
  ignored, `0` is clamped to `1` (`py/_models/_episodes.py:25-41,74-76,110-117`).
- `episode_seed` is fixed at the episode's first predict
  (`_episodes.py:118`).
- An anonymous row (`episode_id == ""`) gets a throwaway context, never
  stored: raw `episode_seed`, `predict_index = 0`, `predict_seed = None`,
  empty state (`_episodes.py:100-107`).
- On an episode end the entry is dropped (its destructors run) before
  `on_episode_end` fires (`_episodes.py:133-137`).
- An evict-all `ResetAdapter` (no ids) or a `ReleaseAdapter` for `env_id`
  drops that env's entries silently, serialized behind in-flight corner calls,
  and fires no `on_episode_end` (MDL-8). An abandoned or timed-out
  `ResetAdapter` (`crates/rlmesh-runtime/src/driver.rs:2087-2106`) still leaks
  until release; the capacity refusal is the backstop.

### Lifetime and threading

- `Model` is a shared handle (R-11): Python lets several sessions drive one
  model, so Rust `Clone`, C `rlmesh_model_clone` and C++ copy share it.
- `run`, `session` and `serve` borrow the model (R-17, X-7). `serve` calls
  `close` when the server stops.
- `close()` is idempotent: the hook fires at most once. It first stops every
  `serve` still running on the model, as a SIGTERM does (drain; that `serve`
  then returns success, and its serve-stop `close` is this one), then waits
  for in-flight corner calls and for calls already queued (a `Session` Drop's
  `on_episode_end` included, R-17), then fires the hook. The drain finishes
  what the server had already received: requests admitted before `close()`
  began run their corner calls normally, and the server admits no request
  after. Every other corner or hook call submitted once `close()` has begun
  fails with the builtin `RuntimeError` class (closed handle) and calls no
  user code: a new `run`, `session` or `serve`, and `predict`, `step` and
  `run` on a Session opened before the close. A `Session` Drop after the
  close submits nothing, so that episode's `on_episode_end` does not fire.
- Dropping the last reference (Rust `Drop`, `rlmesh_model_free`, `~Model`)
  calls `close` if it has not run. Outside an async runtime (always, for C
  and C++) it waits within R-17's Drop bound; inside one (R-16's condition),
  Rust `Drop` never blocks and hands `close` to a detached library thread
  (R-17). Errors are discarded either way; explicit `close()` /
  `close_async()` observes them. The C++ destructor is `noexcept` (MDL-9).
- All corner and hook calls on one Model are serialized across every
  concurrent run, session and serve on it (R-17; MDL-7 for Python), through
  one FIFO call queue per Model. The queue also accepts the `on_episode_end`
  a `Session` Drop submits through the Model's shared internals, ahead of any
  later call (R-17 "Session Drop"). Rust policies take `&mut self`.
- Corners and hooks that the library runs off the caller's thread (behind a
  served or session-driven `Model`) run on plain library threads, never in a
  tokio task or `spawn_blocking`, with the driving flag clear (R-16
  "Library-run callbacks"), so a policy may make sync rlmesh calls (a
  `RemoteModel` query). Which library thread runs a call is unspecified: a
  Rust policy is `Send + 'static` (the trait's supertraits, `pub trait
Policy: Send + 'static`, because the non-generic `Model` erases it and may
  hand its `close` to a detached library thread, MDL-9), C `user_data` and
  state values must be safe to migrate between threads.
- C#: corners and hooks arrive on library threads, a different one from
  call to call, and from several sessions and serves at once, but the call
  queue serializes them per Model (MDL-7), so a C# policy needs no lock for
  state only that Model touches; it does need one for state it shares with
  other code (a Unity component also read by the main thread). Nothing is
  marshalled to a main thread (R-28): a policy that must touch engine
  objects dispatches itself. A policy may make sync rlmesh calls from a
  corner (R-16 "Library-run callbacks"). A corner's exception reaches the
  caller as the same object in process (R-12, X-11).
- A corner or model hook (`load`, `on_episode_end`, `close`; not a run hook,
  which may drive other sessions, RUN-9) must not re-enter its own Model; a
  re-entrant call (a predict, `run`, `session`, `serve`, `close()` or
  `close_async()` on it, or a corner call through a Session on it) fails
  with the builtin `RuntimeError` class instead of deadlocking, in Python too
  (MDL-P3). A last-reference Drop / `rlmesh_model_free` / `~Model` from
  inside a corner or hook never waits for its own call: it hands `close` to
  a detached library thread, as a Drop inside an async runtime does (R-17),
  and that `close` runs once the current call returns. Cancellation uses
  R-18's token, not the model.
- `spec`, `allow_fusion`, `workflow_edition` and the loaded `native_chunk` are
  read once (after `load`) and cached, so getters never wait behind a forward.
  They are always the policy's declarations (or `ModelOptions.spec`); a
  wrapping Python `Model(policy)` reads them from the policy too (MDL-P8).
- Corner errors propagate unchanged (R-12).

### Load

- `load(config)` runs exactly once per Model, eagerly in the constructor:
  `Model::new` / `rlmesh_model_new` / the C++ constructor pass an empty config,
  as `MyModel()` does (`py/_models/base.py:612-618`); `from_config` passes its
  config, as Python's served path does (`py/_bootstrap/loaders.py:348-400`).
  Load errors fail the constructor.
- `load`'s accepted keys are a static declaration the library checks before
  `load` runs, the native form of Python's signature check
  (`loaders.py:380-398`, `params/_resolve.py:66-159`; MDL-19): Rust
  `const LOAD_KEYS: Option<&'static [LoadKey]>`, C `load_keys` +
  `load_keys_len` (R-13's optional list: NULL = None, non-NULL with
  `load_keys_len == 0` = `[]`, written by pointing at any `RLMeshLoadKey`,
  never read; NULL with a nonzero length is an invalid declaration, TypeError
  class), C++ `static constexpr load_keys` (any range of `types::LoadKey`). `LoadKey { name, required }`: `required` is a
  parameter with no default.
  - None (the default; absent in C++): every key reaches `load`, as
    `load(self, **kwargs)`.
  - Some: `from_config` checks, before `load`, in Python's order
    (`params/_resolve.py:136-158`): first a missing required key, refused
    with the ValueError class and `MissingParamError`'s text
    (`missing required parameter: 'x'`, the first missing key in
    declaration order, as Python reports the first in signature order); then
    the keys not listed, refused with the ValueError class and the text
    Python's no-`params` path raises, since natively `params` is deferred
    (MDL-19) and that path resolves against `ParamSpec(extra="passthrough")`
    (`loaders.py:380-381`): `extra='passthrough' but load has no **kwargs to
forward: a, b`, names sorted. Python names the bound method's qualname
    (`Planner.load`, `params/_resolve.py:455-456`); native policies carry no
    type name (C's vtable has none, and C++ wraps C), so the native text
    names `load`. `Model::new` / `rlmesh_model_new` /
    the C++ constructor (empty config) with a required key fail with the
    TypeError class naming it, as `MyModel()` calling `load()` without a
    required argument does. An empty or duplicate name is an invalid
    declaration (TypeError class at construction).
  - Only listed keys are checked; there is no annotation-driven kind check
    natively. A `load` that finds a value of the wrong kind returns the
    ValueError class, as Python's `_check_derived` coercion raises
    `ParamError` (`params/_resolve.py:353-369`).
- The default `load` (not written) accepts only an empty config and refuses a
  nonempty one with the `TypeError` class, naming the keys
  (`loaders.py:388-398`): the Rust default body, a NULL C slot and an absent
  C++ member each refuse it. The text is R-22's, with `model` where Python
  names `{Cls}`: `model received construction params (a, b) but does not
override load(...) to apply them; implement load(**kwargs) to consume the
binding` (names sorted).
- The `rlmesh serve` CLI's `RLMESH_MAKE_KWARGS` feeds `from_config`
  (`py/serve.py:261`); `Model.serve` never reads it.
- Construction order, in every constructor: declarations (corners,
  `context_corners`, static `native_chunk`, `load_keys`) and R-19 sizes are
  checked first, then the config keys, then `load`, then the loaded
  `native_chunk` (any value with no chunk corner is the TypeError class,
  checked first as for a static K; otherwise one below 1 is the ValueError
  class; Fallback rules). On any failure no hook runs (neither `close` nor
  `on_close`) and `*out` is NULL. Per R-20 the library never calls
  `free_user_data` or `free_hook_user_data` on a failed constructor, so the C
  caller still owns `user_data` and `hook_user_data`; Rust drops the policy
  value and the options' closures; the C++ wrapper deletes its trampoline
  object (running the policy's destructor) and throws. Python's `MyModel()`
  checks the declarations only after `load` today (MDL-P7); `from_config`
  already checks them first (`loaders.py:364-400`: `__init__` under
  `suppress_autoload`, then the keys, then `load`).

### Module placement

Rust items are in `rlmesh` (`Policy`, `Model`, `ModelOptions`,
`ModelServeOptions`, `Corners`, `NO_ADAPTER`, `predict_seed`, `with_context`,
`WithContext`)
and `rlmesh::types` (`PredictContext`, `BatchPredictContext`, `SpecArg`, as
in Python's `rlmesh.types`; plus `EpisodeState` and `LoadKey`, native
translations with no Python name, of `PredictContext.state`'s `dict` and of
`load`'s signature, placed beside `PredictContext`); the fenced extras are
`rlmesh::ext::{ModelServeOptions, register_entrypoint}` (R-25). C++ mirrors
them, plus `rlmesh::EpisodeEndHook` beside `ModelOptions`: R-3's "any
callable" for an option Python accepts in two arities. Rust needs no such
type: it has no overloading, so its option takes `|episode_id| …` and a
nullary hook ignores the argument, as its `Policy::on_episode_end(&str)` does.
C# places `Model`, `ModelOptions`, `ModelServeOptions`, `ModelDeclarations`,
`IPolicy` and the corner and hook interfaces in `RLMesh`, `NoAdapter` and
`PredictSeed` on `RLMesh.RLMeshApi` (D-21), `PredictContext`, `SpecArg`,
`EpisodeState` and `LoadKey` in `RLMesh.Types` (Python's and Rust's
`types`), and `Ext.ModelServeOptions`,
`Ext.Entrypoints.Register(string name, ObsTransform transform)` and
`Ext.Entrypoints.Unregister(string name)` (MDL-11, X-16) in `RLMesh.Ext`. Like Rust it needs no `EpisodeEndHook`: `OnEpisodeEnd` is an
`Action<string>`.

### Construction and options

| Python                                                                 | Rust                                                                                                                                                                                                                                                                                | C                                                                                                                                                                                                                                                                                                                                                                                                     | C++                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | C#                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Model(source)`, wrap a callable or policy object                      | `Model::new(policy: impl Policy, options: ModelOptions) -> Result<Model>`; `pub trait Policy: Send + 'static { … }` (Corners, Hooks, Attributes)                                                                                                                                    | `rlmesh_model_new(const RLMeshModelVtable*, void* user_data, const RLMeshModelOptions* /* NULL = INIT */, RLMeshModel** out)`                                                                                                                                                                                                                                                                         | `template<class P, …> explicit Model(P&& policy, ModelOptions options = {})` (`rlmesh::Policy<P>` is a C++17 detection trait; SFINAE selects overloads and a static assertion diagnoses missing predict corners); also `unique_ptr<P>`, `shared_ptr<P>`                                                                                                                                                                                                                                                                                                             | `public Model(IPolicy policy, ModelOptions? options = null)`; `public interface IPolicy { }`, the marker every corner interface extends (Corners, Hooks, Attributes; MDL-23)                                                                                                                                                                                                                                                                                   |
| `Model(lambda obs: ...)`                                               | closures implement `Policy`: `impl<F: FnMut(Value) -> Result<Value> + Send + 'static> Policy for F`                                                                                                                                                                                 | vtable with only `predict`                                                                                                                                                                                                                                                                                                                                                                            | any callable `Value(const Value&)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `public Model(Func<Value, Value> predict, ModelOptions? options = null)`                                                                                                                                                                                                                                                                                                                                                                                       |
| `Model(lambda obs, context: ...)`                                      | `pub fn with_context<F>(f: F) -> WithContext<F> where F: FnMut(Value, &mut PredictContext<'_>) -> Result<Value> + Send + 'static`; `pub struct WithContext<F>(/* private */)`, `impl<F: /* same bound */> Policy for WithContext<F>` (`CORNERS` and `CONTEXT` = `Corners::PREDICT`) | `predict` + `context_corners = RLMESH_CORNER_PREDICT`                                                                                                                                                                                                                                                                                                                                                 | any callable `Value(const Value&, const PredictContext&)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `public Model(Func<Value, PredictContext, Value> predict, ModelOptions? options = null)` (context on `Predict`)                                                                                                                                                                                                                                                                                                                                                |
| Subclass `Model` and override corners                                  | `impl Policy for T` with `CORNERS` / `CONTEXT`                                                                                                                                                                                                                                      | vtable slots (NULL = not defined), `context_corners`                                                                                                                                                                                                                                                                                                                                                  | any class with the members                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | a class implementing `IPolicy`'s corner and hook interfaces (MDL-23)                                                                                                                                                                                                                                                                                                                                                                                           |
| `Model(spec=None)`                                                     | `ModelOptions::spec: Option<SpecArg>`; `None` = `Policy::spec()`                                                                                                                                                                                                                    | `RLMeshModelOptions.spec: RLMeshSpecArg`; `NONE` = the vtable's                                                                                                                                                                                                                                                                                                                                       | `ModelOptions::spec: std::optional<SpecArg>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `ModelOptions.Spec: SpecArg?` (`null` = the declarations')                                                                                                                                                                                                                                                                                                                                                                                                     |
| `Model(on_episode_end=None)` (replaces the policy's hook)              | `ModelOptions::on_episode_end: Option<Box<dyn FnMut(&str) -> Result<()> + Send>>`                                                                                                                                                                                                   | `RLMeshModelOptions.on_episode_end: RLMeshStatus (*)(void* hook_user_data, const char* episode_id)`                                                                                                                                                                                                                                                                                                   | `ModelOptions::on_episode_end: EpisodeEndHook` (empty = the policy's); `class rlmesh::EpisodeEndHook { public: EpisodeEndHook() = default; template<class F> requires std::invocable<F&, std::string_view> \|\| std::invocable<F&> EpisodeEndHook(F f); explicit operator bool() const; void operator()(std::string_view episode_id) const; }`: built from either callable form, as Python accepts `on_episode_end=lambda: ...` (`accepts_episode_id`, `py/_models/base.py:87-116`); the one-argument form wins when callable, as for the policy member (Authoring) | `ModelOptions.OnEpisodeEnd: Action<string>?` (`null` = the policy's; a nullary hook ignores the id, as Rust's)                                                                                                                                                                                                                                                                                                                                                 |
| `Model(on_close=None)` (replaces the policy's `close`; MDL-21, MDL-P6) | `ModelOptions::on_close: Option<Box<dyn FnOnce() -> Result<()> + Send>>`                                                                                                                                                                                                            | `RLMeshModelOptions.on_close: RLMeshStatus (*)(void* hook_user_data)`; `hook_user_data`, `free_hook_user_data`                                                                                                                                                                                                                                                                                        | `ModelOptions::on_close: std::function<void()>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `ModelOptions.OnClose: Action?` (`null` = the policy's `IClose`)                                                                                                                                                                                                                                                                                                                                                                                               |
| `Model(trust_entrypoints=False)`                                       | `ModelOptions::trust_entrypoints: bool` (`false`)                                                                                                                                                                                                                                   | `RLMeshModelOptions.trust_entrypoints` (`false`)                                                                                                                                                                                                                                                                                                                                                      | `ModelOptions::trust_entrypoints = false`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `ModelOptions.TrustEntrypoints: bool` (`false`)                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Model.from_config(**config)`                                          | `Model::from_config::<P: Policy + Default>(config: Kwargs) -> Result<Model>`                                                                                                                                                                                                        | `rlmesh_model_from_config(const RLMeshModelVtable*, void* user_data, const RLMeshValue* config /* DICT */, RLMeshModel** out)`                                                                                                                                                                                                                                                                        | `template<class P> static Model Model::from_config(const Kwargs& config)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `public static Model FromConfig(IPolicy policy, Metadata config)`: `policy` is constructed but not yet loaded; its `Load(config)` runs inside, after the key checks, as C's `rlmesh_model_from_config(vt, user_data, config)` (MDL-23)                                                                                                                                                                                                                         |
| `Model.describe()` (classmethod)                                       | `Model::describe::<P: Policy>() -> Result<Metadata>`                                                                                                                                                                                                                                | `rlmesh_model_vtable_describe(&vt, NULL, &out)`: Core's model type form (COR-10): the vtable's static fields only (non-NULL slots as corners, `spec`, static `native_chunk`, edition, load keys), no slot called, no `user_data`; a vtable `rlmesh_model_new` would refuse is refused with the same status (MDL-13). `rlmesh_model_describe(model, NULL, &out)` is Core's instance form, not this row | `template<class P> static Metadata Model::describe()`: wraps `rlmesh_model_vtable_describe` over the static vtable generated for `P`, constructing nothing (no `load`)                                                                                                                                                                                                                                                                                                                                                                                              | `public static Metadata Describe<TPolicy>(ModelDeclarations? declarations = null) where TPolicy : IPolicy`: corners from the interfaces `TPolicy` implements (a type check, no member reflection), declarations as passed (a `TPolicy` that implements `IModelDeclarations` given `null` is the TypeError class, COR-28); constructs nothing, calls no user code; wraps `rlmesh_model_vtable_describe` and refuses what the constructor would (MDL-13, MDL-23) |
| object reference (R-11)                                                | `impl Clone for Model` (shares)                                                                                                                                                                                                                                                     | `rlmesh_model_clone(const RLMeshModel*, RLMeshModel** out)`                                                                                                                                                                                                                                                                                                                                           | copy shares                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | a C# reference shares (R-11); no `Clone()`                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `rlmesh.types.SpecArg`                                                 | `Option<SpecArg>`; `enum SpecArg { ModelSpec(ModelSpec), NoAdapter }`; `impl From<ModelSpec> for SpecArg` (the Shapes' `model_spec().into()`)                                                                                                                                       | `RLMeshSpecArg { RLMeshSpecKind kind; const RLMeshModelSpec* spec; }`                                                                                                                                                                                                                                                                                                                                 | `std::optional<SpecArg>`; `using SpecArg = std::variant<ModelSpec, NoAdapter>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `SpecArg?`; `public sealed class SpecArg` with `ModelSpec? ModelSpec`, `bool IsNoAdapter` and `implicit operator SpecArg(ModelSpec spec)`                                                                                                                                                                                                                                                                                                                      |
| `NO_ADAPTER`                                                           | `pub const NO_ADAPTER: SpecArg = SpecArg::NoAdapter`                                                                                                                                                                                                                                | `RLMESH_SPEC_KIND_NO_ADAPTER`                                                                                                                                                                                                                                                                                                                                                                         | `inline constexpr NoAdapter NO_ADAPTER{}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `public static readonly SpecArg NoAdapter` on `RLMesh.RLMeshApi` (R-5, D-21)                                                                                                                                                                                                                                                                                                                                                                                   |
| `rlmesh.types.Value`                                                   | [Core](core.md)                                                                                                                                                                                                                                                                     | [Core](core.md)                                                                                                                                                                                                                                                                                                                                                                                       | [Core](core.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | [Core](core.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                |

### Corners

| Python                                                              | Rust (`Policy`)                                                                                                                                                                            | C (`RLMeshModelVtable`)                                                                                                                                              | C++ (detected member)                                                                                  | C# (interface)                                                                                                                                                                                                                  |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `predict(observation[, context])`                                   | `fn predict(&mut self, observation: Value) -> Result<Value>`; `fn predict_with_context(&mut self, observation: Value, context: &mut PredictContext<'_>) -> Result<Value>`                  | `RLMeshStatus (*predict)(void*, const RLMeshValue* observation, const RLMeshPredictContext* context, RLMeshValue** out_action)`                                      | `Value predict(const Value&[, const PredictContext&])`                                                 | `IPredict`: `Value Predict(Value observation)`; `IPredictWithContext`: `Value Predict(Value observation, PredictContext context)`                                                                                               |
| `predict_batch(observations[, context])`                            | `fn predict_batch(&mut self, observations: Vec<Value>) -> Result<Vec<Value>>`; `predict_batch_with_context(.., context: &mut BatchPredictContext<'_>)`                                     | `RLMeshStatus (*predict_batch)(void*, const RLMeshValue* const* observations, size_t count, const RLMeshPredictContext* const* contexts, RLMeshValue** out_actions)` | `std::vector<Value> predict_batch(rlmesh::Span<const Value>[, BatchPredictContext])`                   | `IPredictBatch`: `IReadOnlyList<Value> PredictBatch(IReadOnlyList<Value> observations)`; `IPredictBatchWithContext`: `PredictBatch(IReadOnlyList<Value> observations, IReadOnlyList<PredictContext> context)`                   |
| `predict_chunk(observation[, execution_horizon][, context])`        | `fn predict_chunk(&mut self, observation: Value, execution_horizon: u32) -> Result<Value>`; `predict_chunk_with_context(.., context: &mut PredictContext<'_>)`                             | `RLMeshStatus (*predict_chunk)(void*, const RLMeshValue*, uint32_t execution_horizon, const RLMeshPredictContext*, RLMeshValue** out_chunk)`                         | `Value predict_chunk(const Value&[, uint32_t execution_horizon][, const PredictContext&])`             | `IPredictChunk`: `Value PredictChunk(Value observation, uint executionHorizon)`; `IPredictChunkWithContext`: `PredictChunk(Value observation, uint executionHorizon, PredictContext context)`                                   |
| `predict_chunk_batch(observations[, execution_horizon][, context])` | `fn predict_chunk_batch(&mut self, observations: Vec<Value>, execution_horizon: u32) -> Result<Vec<Value>>`; `predict_chunk_batch_with_context(.., context: &mut BatchPredictContext<'_>)` | `RLMeshStatus (*predict_chunk_batch)(void*, const RLMeshValue* const*, size_t, uint32_t, const RLMeshPredictContext* const*, RLMeshValue** out_chunks)`              | `std::vector<Value> predict_chunk_batch(rlmesh::Span<const Value>[, uint32_t][, BatchPredictContext])` | `IPredictChunkBatch`: `IReadOnlyList<Value> PredictChunkBatch(IReadOnlyList<Value> observations, uint executionHorizon)`; `IPredictChunkBatchWithContext`: `(.., uint executionHorizon, IReadOnlyList<PredictContext> context)` |
| override detection (`_overridden`, `accepts_context`)               | `const CORNERS: Corners = Corners::PREDICT`; `const CONTEXT: Corners = Corners::NONE`                                                                                                      | non-NULL slots; `uint32_t context_corners` (`RLMESH_CORNER_PREDICT` 1, `_PREDICT_CHUNK` 2, `_PREDICT_BATCH` 4, `_PREDICT_CHUNK_BATCH` 8)                             | compile-time member detection                                                                          | implemented interfaces, read with `is` once at construction; one corner implemented in both forms is refused (TypeError class, MDL-23)                                                                                          |
| corner fallback (`_synthesize_corners`) and route-kind dispatch     | the rules above, applied by `Model`                                                                                                                                                        | the same, applied by `rlmesh_model_new` and the engine                                                                                                               | the same, through C                                                                                    | the same, through C                                                                                                                                                                                                             |

`Corners` is a bit set with C's bit values:

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Corners(u32); // private field; Default = NONE
impl Corners {
    pub const NONE: Corners = Corners(0);
    pub const PREDICT: Corners = Corners(1);
    pub const PREDICT_CHUNK: Corners = Corners(2);
    pub const PREDICT_BATCH: Corners = Corners(4);
    pub const PREDICT_CHUNK_BATCH: Corners = Corners(8);
    pub const fn union(self, other: Corners) -> Corners;      // bitwise or
    pub const fn contains(self, other: Corners) -> bool;      // every bit of other is in self
}
```

`execution_horizon` is `u32`: the wire field is `uint32`
(`crates/rlmesh-proto/proto/rlmesh/model/v1/interaction.proto:61`) and Python
caps it at 1,024 (`py/_models/_eval.py:70,775-780`). Default corner bodies in
Rust return `Error::Runtime` (`ErrorClass::RuntimeError`; Python's
`NotImplementedError`, folded by R-21, `py/_models/base.py:785-851`), with
text naming the missing `CORNERS` / `CONTEXT` entry (R-22). C and C++ have no
such path: a NULL slot or an undetected member is never called, so an
undefined corner cannot be reached there (Authoring).

### Hooks and lifecycle

| Python                                   | Rust                                                                                                                                                                                                                                                | C                                                                                                                                                                             | C++                                                                                                                                                                                                                                                                                          | C#                                                                                                                                                                                                                                                 |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Model.on_episode_end(episode_id='')`    | `fn on_episode_end(&mut self, episode_id: &str) -> Result<()> { Ok(()) }`                                                                                                                                                                           | `RLMeshStatus (*on_episode_end)(void*, const char* episode_id)`                                                                                                               | `void on_episode_end(std::string_view episode_id)` or `void on_episode_end()` (Authoring)                                                                                                                                                                                                    | `IOnEpisodeEnd`: `void OnEpisodeEnd(string episodeId)` (MDL-23)                                                                                                                                                                                    |
| `Model.close()` (authored)               | `fn close(&mut self) -> Result<()> { Ok(()) }`                                                                                                                                                                                                      | `RLMeshStatus (*close)(void*)`                                                                                                                                                | `void close()`                                                                                                                                                                                                                                                                               | `IClose`: `void Close()`; not `IDisposable` (MDL-23)                                                                                                                                                                                               |
| `Model.close()` (handle)                 | `Model::close(&self) -> Result<()>`; `close_async`                                                                                                                                                                                                  | `RLMeshStatus rlmesh_model_close(RLMeshModel*)`                                                                                                                               | `void Model::close()`                                                                                                                                                                                                                                                                        | `public void Close()`; `public Task CloseAsync(CancellationToken cancellationToken = default)` (R-16)                                                                                                                                              |
| (none: explicit `close()` only; MDL-9)   | `Drop` of the last clone closes if not closed                                                                                                                                                                                                       | `rlmesh_model_free` (last reference closes); NULL no-op                                                                                                                       | `~Model() noexcept` (last copy closes)                                                                                                                                                                                                                                                       | `Dispose()` / `using`: `rlmesh_model_free` once every session's reference on its ownership record is released (R-17, R-28), closing if not closed, errors discarded; a finalized `Model` frees on R-28's cleanup thread                            |
| `Model.load(**kwargs)`                   | `fn load(&mut self, config: &Kwargs) -> Result<()>` (default: empty only)                                                                                                                                                                           | `RLMeshStatus (*load)(void*, const RLMeshValue* config /* DICT */)`; NULL = empty only                                                                                        | `void load(const Kwargs& config)`, or `void load()` (= `load_keys = {}`, Authoring); absent = empty only                                                                                                                                                                                     | `ILoad`: `void Load(Metadata config)`; absent = empty only; Python's nullary `load(self)` is `ILoad` with `LoadKeys = []` (MDL-23)                                                                                                                 |
| `load`'s signature (keys, defaults)      | `const LOAD_KEYS: Option<&'static [LoadKey]> = None`; `pub struct LoadKey { pub name: &'static str, pub required: bool }`, `const fn LoadKey::optional(name: &'static str) -> LoadKey`, `const fn LoadKey::required(name: &'static str) -> LoadKey` | vtable `const RLMeshLoadKey* load_keys; size_t load_keys_len;` (R-13: NULL = None, non-NULL with 0 = `[]`; deep-copied); `RLMeshLoadKey { const char* name; bool required; }` | `static constexpr` range of `types::LoadKey { std::string_view name; bool required = false; }` named `load_keys` (absent = None)                                                                                                                                                             | `ModelDeclarations.LoadKeys: IReadOnlyList<LoadKey>?` (`null` = None); `public sealed record LoadKey(string Name) { public bool Required { get; init; } }` in `RLMesh.Types`                                                                       |
| `Model.serve(address, *, options=None)`  | `fn serve(&self, address: &str, options: ModelServeOptions) -> Result<()>`; `serve_async`                                                                                                                                                           | `RLMeshStatus rlmesh_model_serve(RLMeshModel*, const char* address, const RLMeshModelServeOptions* /* NULL = INIT */)`                                                        | `void serve(std::string_view address, const ModelServeOptions& options = {})`                                                                                                                                                                                                                | `public void Serve(string address, ModelServeOptions? options = null)`; `public Task ServeAsync(string address, ModelServeOptions? options = null, CancellationToken cancellationToken = default)` (MDL-24)                                        |
| `options: ServeOptions \| None = None`   | `ModelServeOptions::options: Option<ServeOptions>` ([Env](env.md))                                                                                                                                                                                  | `RLMeshModelServeOptions.options: const RLMeshServeOptions*` (NULL = None)                                                                                                    | `ModelServeOptions::options: std::optional<ServeOptions>`                                                                                                                                                                                                                                    | `ModelServeOptions.Options: ServeOptions?` ([Env](env.md))                                                                                                                                                                                         |
| (KeyboardInterrupt, R-18)                | `ModelServeOptions::cancel: Option<ext::CancelToken>`                                                                                                                                                                                               | `RLMeshExtCancelToken* cancel` (NULL = none, borrowed for the call)                                                                                                           | `std::optional<ext::CancelToken> cancel;`                                                                                                                                                                                                                                                    | `ModelServeOptions.Cancel: Ext.CancelToken?` (R-18)                                                                                                                                                                                                |
| (language extras, MDL-20)                | `ModelServeOptions::ext: Option<ext::ModelServeOptions>`; `ext::ModelServeOptions { pub token: Option<String>, pub predict_concurrency: Option<usize> }`                                                                                            | `const RLMeshExtModelServeOptions* ext` (NULL = none)                                                                                                                         | `std::optional<ext::ModelServeOptions> ext`; `ext::ModelServeOptions { std::optional<std::string> token; std::optional<size_t> predict_concurrency; bool no_signal_handlers = false; }` (the last pending X-15)                                                                              | `ModelServeOptions.Ext: Ext.ModelServeOptions?`; `public sealed class Ext.ModelServeOptions { string? Token; int? PredictConcurrency; bool NoSignalHandlers = false; }` (`int`: an in-memory count, D-34; `NoSignalHandlers` pending X-15, MDL-24) |
| `Model.run`, `Model.session`             | [Running](running.md)                                                                                                                                                                                                                               | [Running](running.md)                                                                                                                                                         | [Running](running.md)                                                                                                                                                                                                                                                                        | [Running](running.md)                                                                                                                                                                                                                              |
| `RANDOM_SAMPLE`                          | [Running](running.md)                                                                                                                                                                                                                               | [Running](running.md)                                                                                                                                                         | [Running](running.md)                                                                                                                                                                                                                                                                        | [Running](running.md)                                                                                                                                                                                                                              |
| duck-typed `reset()` on a wrapped object | not carried over (MDL-10)                                                                                                                                                                                                                           | not carried over                                                                                                                                                              | `static_assert` on any member named `reset`, of any signature, whether or not `on_episode_end` is present, as Python refuses any subclass `reset`: detected when `&P::reset` is well-formed (one member), or when `p.reset()` or `p.reset(std::string_view{})` is (an overload set) (MDL-10) | not carried over, and nothing refused: C# reads no member it has no interface for (R-28), as Rust and C (MDL-10)                                                                                                                                   |

### Serve

`Model.serve` blocks until the server stops, then calls `close` (R-17). Its
rules are Python's (`py/_models/base.py:1475-1500`,
`pyrs/model.rs:875-925`; MDL-20):

- **Options.** `options` is Env's shared `ServeOptions`, validated by `serve`
  (R-6). `None` drain and close timeouts take Env's 5 s caps (ENV-7), as
  Python's model serve does (`pyrs/model.rs:84,888-891`), applied in the
  facade so every binding gets them (Rust and C current differences).
- **Edition.** The endpoint declares the resolved workflow edition (COR-6;
  `options.workflow_edition` is the call's value, COR-P2) on every
  handshake, and warns `WorkflowEditionWarning` when an authored policy's
  edition floats (R-23, COR-7; `py/_models/base.py:1491-1501`,
  `py/_editions.py:251-282`).
- **Signals** follow ENV-11 for a foreground sync `serve`: SIGINT stops the
  server (drain, then `close`) and `serve` fails with R-21's cancellation
  class; a second SIGINT behaves as ENV-11 says (natively: the default
  SIGINT disposition is restored and the signal re-raised). SIGTERM stops it
  the same way and `serve` returns success. `serve_async` installs no signal
  handlers and resolves when the server stops, as ENV-29 does for env
  servers; its caller cancels it with `cancel` or by dropping the future.
- **Cancel.** Cancelling `cancel` (R-18) acts as SIGINT: drain, `close`, then
  the cancellation class (`Error::Cancelled`, `RLMESH_ERR_CANCELLED`,
  `rlmesh::ext::CancelledError`). The model is closed afterwards, so later
  `run`, `session` and `serve` fail with the builtin `RuntimeError` class. A
  token cancelled before `serve` starts fails it at once, before binding, and
  does not close the model. Dropping a `serve_async` future after its first
  poll is a cancellation with the same model state (R-18): it stops the
  server and never blocks the dropping task; the drain and the serve-stop
  `close` finish detached on a library thread, as a Drop inside a runtime
  hands `close` off (MDL-9) and as a dropped `EnvServer::serve_async` closes
  its env (ENV-32), and their errors are discarded. `close` is a model hook
  run on library threads (R-16 "Library-run callbacks"), not one of the
  caller-dispatched run hooks R-18 suppresses on drop. The model is closed
  afterwards, as after a token cancel: later `run`, `session` and `serve`
  fail with the builtin `RuntimeError` class, and an explicit `close()`
  waits for the detached close, fires no hook again (Lifetime) and returns
  success, the discarded error included. A future dropped before
  its first poll has done nothing and does nothing (ENV-32).
- **Endpoint token.** `RLMESH_MODEL_ENDPOINT_TOKEN`, when set, requires that
  bearer token on every model RPC and wins over `ext.token` (which is then
  not checked). The classes split as in Env's construction step 8: a set
  variable that is empty, whitespace-only, not Unicode or cannot travel in an
  HTTP header fails startup before bind with the `RLMeshException` class
  (`Error::Internal`, `RLMESH_ERR_INTERNAL`), as Python's does
  (`base.py:1487-1489`, `rs/serve_options.rs:141-170`). With the variable
  unset, an `ext.token` that cannot travel in an HTTP header is an invalid
  options field (R-6), so it fails before bind with the `ValueError` class
  (`Error::InvalidArgument`, `RLMESH_ERR_INVALID_ARGUMENT`,
  `rlmesh::ValueError`; R-21). A valid nonempty `ext.token` becomes the
  required bearer token; `None`, NULL or `""` serves without authentication.
  Python has no `ext`; it reads only the variable.
- **`ext.predict_concurrency`** caps the Join-stream requests the served model
  processes at once per connection; `None` is the build's default cap
  (`rlmesh_grpc::DEFAULT_PREDICT_CONCURRENCY`), and `Some(0)` fails with the
  ValueError class. Corner calls stay serialized (R-17); the cap bounds
  decode, encode and the critical sections that overlap them.

### Attributes

| Python                                                                                                                                                                    | Rust                                                                                                                                                                                       | C                                                                                                                                                                                                                                      | C++                                                                                                                                                                                               | C#                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Model.spec = None` (class)                                                                                                                                               | `fn spec() -> Option<SpecArg> { None }` (associated, no `self`)                                                                                                                            | vtable `RLMeshSpecArg spec` (`{NONE, NULL}`)                                                                                                                                                                                           | `static SpecArg spec()` (absent = None)                                                                                                                                                           | `ModelDeclarations.Spec: SpecArg?` (`null`)                                                                                                                                |
| `Model.spec` (resolved)                                                                                                                                                   | `Model::spec(&self) -> Option<&SpecArg>`                                                                                                                                                   | `RLMeshStatus rlmesh_model_spec(const RLMeshModel*, RLMeshSpecArg* out)`                                                                                                                                                               | `const std::optional<SpecArg>& Model::spec() const`                                                                                                                                               | `public SpecArg? Spec { get; }`                                                                                                                                            |
| `Model.allow_fusion = True` (a wrapping `Model(policy)` reads the policy's, MDL-P8)                                                                                       | `fn allow_fusion(&self) -> bool { true }` (read after load); `Model::allow_fusion(&self) -> bool`                                                                                          | vtable `bool allow_fusion` (INIT `true`); `bool rlmesh_model_allow_fusion(const RLMeshModel*)`                                                                                                                                         | `static constexpr bool allow_fusion` (absent = true), read at construction (MDL-18); `bool Model::allow_fusion() const`                                                                           | `ModelDeclarations.AllowFusion: bool` (`true`), read at construction (MDL-18); `public bool AllowFusion { get; }`                                                          |
| `Model.native_chunk = None` (wrapping: the policy's, MDL-P8)                                                                                                              | `const NATIVE_CHUNK: Option<u32> = None` (static, describe); `fn native_chunk(&self) -> Option<u32> { Self::NATIVE_CHUNK }` (read after load); `Model::native_chunk(&self) -> Option<u32>` | vtable `bool has_native_chunk` + `uint32_t native_chunk` (static); slot `loaded_native_chunk(void*, bool* out_has, uint32_t* out)` (NULL = static); `void rlmesh_model_native_chunk(const RLMeshModel*, bool* out_has, uint32_t* out)` | `static constexpr std::optional<uint32_t> native_chunk`; optional member `std::optional<uint32_t> loaded_native_chunk()` (read after load); `std::optional<uint32_t> Model::native_chunk() const` | `ModelDeclarations.NativeChunk: uint?` (static, describe); `ILoadedNativeChunk`: `uint? LoadedNativeChunk { get; }` (read after load); `public uint? NativeChunk { get; }` |
| `Model.workflow_edition = None` (a wrapping `Model(policy)` reads `type(policy).workflow_edition` and is authored unless it wraps a bare callable, COR-P6, COR-7; MDL-P8) | `fn workflow_edition() -> Option<String> { None }` (associated, no `self`; ENV-14); `Model::workflow_edition(&self) -> Option<&str>`                                                       | vtable `const char* workflow_edition` (NULL = None, deep-copied); `RLMeshStatus rlmesh_model_workflow_edition(const RLMeshModel*, bool* out_has, RLMeshBytes* out)` (R-5; `out` untouched when `*out_has` is false)                    | `static std::optional<std::string> workflow_edition()` (absent = None); `std::optional<std::string> Model::workflow_edition() const`                                                              | `ModelDeclarations.WorkflowEdition: string?`; `public string? WorkflowEdition { get; }`                                                                                    |
| `Model.params`                                                                                                                                                            | deferred (Experimental)                                                                                                                                                                    | deferred                                                                                                                                                                                                                               | deferred                                                                                                                                                                                          | deferred                                                                                                                                                                   |
| `Model.device`                                                                                                                                                            | none (bridge, R-24)                                                                                                                                                                        | none                                                                                                                                                                                                                                   | none                                                                                                                                                                                              | none                                                                                                                                                                       |

### Context and seeds

| Python                                       | Rust                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | C                                                                                           | C++                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | C#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rlmesh.types.PredictContext`                | `rlmesh::types::PredictContext<'a>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `RLMeshPredictContext` (library-filled, read-only, valid for the call)                      | `rlmesh::types::PredictContext`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `RLMesh.Types.PredictContext`, a sealed class lent for the corner call (MDL-25)                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `.episode_id: str`                           | `episode_id: &'a str`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `const char* episode_id` (`""` anonymous)                                                   | `std::string_view episode_id`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `string EpisodeId { get; }` (`""` anonymous)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `.episode_seed: int \| None`                 | `episode_seed: Option<i64>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `bool has_episode_seed; int64_t episode_seed`                                               | `std::optional<int64_t> episode_seed`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `long? EpisodeSeed { get; }`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `.predict_index: int`                        | `predict_index: u64`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `uint64_t predict_index`                                                                    | `uint64_t predict_index`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `ulong PredictIndex { get; }`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `.predict_seed: int \| None`                 | `predict_seed: Option<i64>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `bool has_predict_seed; int64_t predict_seed`                                               | `std::optional<int64_t> predict_seed`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `long? PredictSeed { get; }`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `.state: dict[str, Any]`                     | `state: &'a mut EpisodeState` (`Box<dyn Any + Send>` values): `fn get<T: Any>(&self, key: &str) -> Option<&T>`, `fn get_mut<T: Any>(&mut self, key: &str) -> Option<&mut T>` (`None` when missing or another type); `fn insert<T: Any + Send>(&mut self, key: impl Into<String>, value: T) -> Option<Box<dyn Any + Send>>` (the replaced value); `fn get_or_insert_with<T: Any + Send>(&mut self, key: &str, f: impl FnOnce() -> T) -> Result<&mut T>` (`Error::Type` naming the key when it holds another type); `fn remove(&mut self, key: &str) -> Option<Box<dyn Any + Send>>`; `fn contains_key(&self, key: &str) -> bool`; `fn len(&self) -> usize`; `fn is_empty(&self) -> bool` | `RLMeshEpisodeState* state`; `rlmesh_episode_state_get/set/remove/len` (declarations below) | `EpisodeState& state`: `template<class T> T* get(std::string_view key)`, `template<class T> const T* get(std::string_view key) const` (`nullptr` when missing or another `T`); `template<class T> void insert_or_assign(std::string_view key, T value)` (replaces any value); `template<class T, class... A> T& get_or_emplace(std::string_view key, A&&... args)` (throws `rlmesh::TypeError` when the key holds another type); `bool erase(std::string_view key)` (false when missing); `bool contains_key(std::string_view key) const`; `size_t size() const`; `bool empty() const`. The dict operations take Core's `Metadata` names in both languages (Rust `contains_key` / `insert` / `remove` / `len` / `is_empty`, C++ `contains_key` / `insert_or_assign` / `erase` / `size` / `empty`; [Core](core.md) Values) | `EpisodeState State { get; }`: `bool TryGetValue<T>(string key, out T value)` (false when missing or another type); `T GetOrAdd<T>(string key, Func<T> create)` (TypeError class when the key holds another type); `object? this[string key] { get; set; }` (get on a missing key: `KeyNotFoundException`; set replaces); `bool Remove(string key)`; `bool ContainsKey(string key)`; `int Count`. Values are any `object?`, held through a `GCHandle` passed to `rlmesh_episode_state_set` with a static `free_value` (MDL-25) |
| `BatchPredictContext = list[PredictContext]` | `type BatchPredictContext<'a> = [PredictContext<'a>]`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `const RLMeshPredictContext* const*` of `count`                                             | `using BatchPredictContext = rlmesh::Span<const PredictContext>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `IReadOnlyList<PredictContext>` (no alias type, D-30)                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `predict_seed(episode_seed, predict_index)`  | `fn predict_seed(episode_seed: i64, predict_index: u64) -> i64`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `int64_t rlmesh_predict_seed(int64_t episode_seed, uint64_t predict_index)`                 | `int64_t predict_seed(int64_t, uint64_t)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `public static long PredictSeed(long episodeSeed, ulong predictIndex)` on `RLMesh.RLMeshApi`                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| episode store (65,536, refuse, env drop)     | as above; `Error::Runtime`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | as above; `RLMESH_ERR_RUNTIME`                                                              | as above; `rlmesh::RuntimeError`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | as above; `RLMesh.RuntimeError`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

### Target declarations (C)

```c
typedef enum RLMeshSpecKind {
  RLMESH_SPEC_KIND_INVALID = 0, RLMESH_SPEC_KIND_NONE = 1,
  RLMESH_SPEC_KIND_MODEL_SPEC = 2, RLMESH_SPEC_KIND_NO_ADAPTER = 3
} RLMeshSpecKind;
typedef struct RLMeshSpecArg {          /* by value, frozen per ABI generation (R-19) */
  RLMeshSpecKind kind;
  const RLMeshModelSpec* spec;          /* MODEL_SPEC only; deep-copied by the consumer */
} RLMeshSpecArg;

#define RLMESH_CORNER_PREDICT 1u
#define RLMESH_CORNER_PREDICT_CHUNK 2u
#define RLMESH_CORNER_PREDICT_BATCH 4u
#define RLMESH_CORNER_PREDICT_CHUNK_BATCH 8u

typedef struct RLMeshEpisodeState RLMeshEpisodeState;  /* borrowed, valid for the call */
/* true and *out = value when present (a stored NULL is a value); false when missing. */
RLMESH_API bool rlmesh_episode_state_get(const RLMeshEpisodeState*, const char* key, void** out);
/* R-20: on RLMESH_OK value (and free_value, NULL = none) pass to the state, and replacing
 * a key frees the old value first; on nonzero free_value is not called, the caller still
 * owns value, and the state is unchanged. */
RLMESH_API RLMeshStatus rlmesh_episode_state_set(RLMeshEpisodeState*, const char* key, void* value,
                                                 void (*free_value)(void* value));
RLMESH_API bool rlmesh_episode_state_remove(RLMeshEpisodeState*, const char* key);
RLMESH_API size_t rlmesh_episode_state_len(const RLMeshEpisodeState*);

typedef struct RLMeshPredictContext {   /* struct_size = the library's sizeof */
  size_t struct_size;
  const char* episode_id;
  bool has_episode_seed; int64_t episode_seed;
  uint64_t predict_index;
  bool has_predict_seed; int64_t predict_seed;
  RLMeshEpisodeState* state;
} RLMeshPredictContext;
/* Lent (R-19): a reader treats any field struct_size does not cover as this value. */
#define RLMESH_PREDICT_CONTEXT_INIT {sizeof(RLMeshPredictContext), "", false, 0, 0, false, 0, NULL}

typedef struct RLMeshLoadKey {   /* by value, frozen per ABI generation (R-19) */
  const char* name;              /* nonempty, unique */
  bool required;                 /* no default in load */
} RLMeshLoadKey;

/* Corners write OWNED values into pre-zeroed out slots. On nonzero the library
 * frees rows already written. context(s) is NULL unless the corner's bit is set
 * in context_corners. */
typedef struct RLMeshModelVtable {
  size_t struct_size;
  RLMeshStatus (*predict)(void* user_data, const RLMeshValue* observation,
                          const RLMeshPredictContext* context, RLMeshValue** out_action);
  RLMeshStatus (*predict_chunk)(void* user_data, const RLMeshValue* observation, uint32_t execution_horizon,
                                const RLMeshPredictContext* context, RLMeshValue** out_chunk);
  RLMeshStatus (*predict_batch)(void* user_data, const RLMeshValue* const* observations, size_t count,
                                const RLMeshPredictContext* const* contexts, RLMeshValue** out_actions);
  RLMeshStatus (*predict_chunk_batch)(void* user_data, const RLMeshValue* const* observations, size_t count,
                                      uint32_t execution_horizon, const RLMeshPredictContext* const* contexts,
                                      RLMeshValue** out_chunks);
  RLMeshStatus (*on_episode_end)(void* user_data, const char* episode_id);
  RLMeshStatus (*close)(void* user_data);
  RLMeshStatus (*load)(void* user_data, const RLMeshValue* config);
  RLMeshStatus (*loaded_native_chunk)(void* user_data, bool* out_has, uint32_t* out);
  void (*free_user_data)(void* user_data);
  uint32_t context_corners;      /* 0; bits of non-NULL slots only, else RLMESH_ERR_TYPE */
  RLMeshSpecArg spec;            /* {RLMESH_SPEC_KIND_NONE, NULL} */
  bool allow_fusion;             /* true */
  bool has_native_chunk;         /* false */
  uint32_t native_chunk;         /* static K for describe; 0 with has_native_chunk: ValueError class */
  const char* workflow_edition;  /* NULL = None */
  const RLMeshLoadKey* load_keys;  /* NULL = None: every key reaches load (MDL-19); R-13 */
  size_t load_keys_len;            /* entries, deep-copied; non-NULL with 0 = []; NULL with >0: RLMESH_ERR_TYPE */
} RLMeshModelVtable;
#define RLMESH_MODEL_VTABLE_INIT {sizeof(RLMeshModelVtable), NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
                                  NULL, NULL, 0, {RLMESH_SPEC_KIND_NONE, NULL}, true, false, 0, NULL, \
                                  NULL, 0}


typedef struct RLMeshModelOptions {
  size_t struct_size;
  RLMeshSpecArg spec;                                                            /* NONE = vtable's */
  RLMeshStatus (*on_episode_end)(void* hook_user_data, const char* episode_id); /* NULL = vtable's */
  RLMeshStatus (*on_close)(void* hook_user_data);                                /* NULL = vtable's close */
  void* hook_user_data;
  void (*free_hook_user_data)(void* hook_user_data);
  bool trust_entrypoints;                                                        /* false */
  uint8_t _reserved[sizeof(size_t) - 1];
} RLMeshModelOptions;
#define RLMESH_MODEL_OPTIONS_INIT {sizeof(RLMeshModelOptions), {RLMESH_SPEC_KIND_NONE, NULL}, NULL, NULL, \
                                   NULL, NULL, false, {0}}

typedef struct RLMeshExtModelServeOptions {      /* fenced (R-25), MDL-20 */
  size_t struct_size;
  const char* token;                             /* NULL/"" = none; RLMESH_MODEL_ENDPOINT_TOKEN wins */
  size_t predict_concurrency;                    /* 0 with has_: ValueError class */
  bool has_predict_concurrency;                  /* false = the default cap */
  uint8_t _reserved[sizeof(size_t) - 1];
  /* Pending X-15 (index): appends bool no_signal_handlers (false = install, ENV-11)
   * and uint8_t _reserved1[sizeof(size_t) - 1]. */
} RLMeshExtModelServeOptions;
#define RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT {sizeof(RLMeshExtModelServeOptions), NULL, 0, false, {0}}

typedef struct RLMeshModelServeOptions {
  size_t struct_size;
  const RLMeshServeOptions* options;             /* NULL = None (Env) */
  RLMeshExtCancelToken* cancel;                  /* NULL = none, borrowed (R-18) */
  const RLMeshExtModelServeOptions* ext;         /* NULL = none */
} RLMeshModelServeOptions;
#define RLMESH_MODEL_SERVE_OPTIONS_INIT {sizeof(RLMeshModelServeOptions), NULL, NULL, NULL}

RLMESH_API RLMeshStatus rlmesh_model_serve(RLMeshModel* model, const char* address,
                                           const RLMeshModelServeOptions* options /* NULL = INIT */);
/* Fenced (R-25), MDL-11. Takes its own reference to transform only on RLMESH_OK (ADP-7);
 * the caller frees its handle when it likes. A replaced name's reference is released. */
RLMESH_API RLMeshStatus rlmesh_ext_register_entrypoint(const char* name, const RLMeshObsTransform* transform);
```

`rlmesh_model_new` copies the vtable and options and deep-copies `spec`,
`workflow_edition` and `load_keys`; on `RLMESH_OK`, `user_data` and
`hook_user_data` pass to the model and are released through their `free_*`
when the last reference is freed. On any nonzero return the library has
called neither `free_*`, nor `close` / `on_close`, `*out` is NULL, and the
caller still owns both (R-20; construction order under Load). Getter results
live until the last reference is freed. `rlmesh_model_serve` reads its
options for the call only. Sizing follows R-19 (`RLMESH_MODEL_VTABLE_MIN_SIZE`,
`RLMESH_MODEL_OPTIONS_MIN_SIZE`, `RLMESH_PREDICT_CONTEXT_MIN_SIZE`,
`RLMESH_MODEL_SERVE_OPTIONS_MIN_SIZE`,
`RLMESH_EXT_MODEL_SERVE_OPTIONS_MIN_SIZE`).

## Python changes

- **MDL-P1** An authored subclass `close()` runs at most once. Today the
  `__init_subclass__` wrapper has no guard and re-runs it on every call
  (`py/_models/base.py:853-866`). Justification: X-7, idempotent close.
- **MDL-P2** `close()` is terminal (Lifetime): it stops a `serve` still
  running on the model (drain: requests it already received finish; that
  `serve` returns), waits for in-flight corner calls, and every other corner
  or hook call submitted after it began raises `RuntimeError`
  without calling user code: new `run`, `session` and `serve`, and
  `predict`, `step` and `run` on Sessions opened before the close; a Session
  closed afterwards fires no `on_episode_end`. Today `close()`
  only fires the hook and drops `_worker` (`py/_models/base.py:884-899`), the
  next serve rebuilds the worker (`:914-915`), and `session` calls the raw
  corners directly (`:1449-1473`), so a closed model keeps working.
  Justification: X-7.
- **MDL-P3** Corner and hook calls on one Model are serialized across
  concurrent local sessions and runs, as the native worker already does. Today
  a local `Session` calls the corner directly (`py/_models/_eval.py:1110-1145`),
  so two threaded sessions can overlap, and a corner may drive another session
  on its own model. With calls serialized, a corner or hook that re-enters its
  own Model (a predict, `run`, `session`, `serve` or `close()` on it, or a
  corner call through a Session on it) raises `RuntimeError` instead of
  deadlocking
  (Lifetime). Justification: R-17; MDL-7 (User (accepted default)).
- **MDL-P4** Episode-store entries are tagged with `env_id`, and an evict-all
  `ResetAdapter` or a `ReleaseAdapter` drops that env's entries without firing
  `on_episode_end`. Today the store keys only by `episode_id`
  (`py/_models/_episodes.py:62-137`) and the engine returns early on empty ids
  (`rs/model/engine.rs:1209-1221`), so sessions ended by teardown leak until
  capacity. Justification: MDL-8 (User (accepted default)).
- **MDL-P5** `native_chunk` is read once after `load` and cached; a value
  below 1 is refused there with its own `ValueError`. Today it is read live
  off the policy on each use (`py/_models/base.py:742-750`) and checked only at
  worker build (`:1060-1064`); a session refuses a K below 1 only indirectly,
  because the horizon is already `>= 1` and so exceeds K, with the horizon
  message (`py/_models/_eval.py:775-788`). The refusal stays; its read point
  and message change. A K set in `load()` on a model with no chunk corner is
  refused at that same read point with `TypeError`, as a static K is
  (`base.py:728-740`). Today the static check reads only the class (`:732`), so
  `Model.run` and `serve` refuse it only at route resolve with
  `RuntimeError("model error: …")` (`rs/model/engine.rs:1283-1291`,
  `pyrs/types.rs:50`), and a local Session never refuses it. Changelog
  entry: the late `RuntimeError` becomes a `TypeError` from the constructor
  or `from_config`. Justification: one read point for the engine and
  getters that never wait behind a forward; one class for a static and a
  loaded K (Authoring).
- **MDL-P6** `on_close=` replaces a subclass's authored `close()` body too,
  as the constructor docstring promises (`py/_models/base.py:453-455`): with
  `on_close` set, `model.close()` runs the hook once instead of the authored
  body. Today a subclass's authored `close()` stays the public close
  (`coerced_on_close = None`, `:621-624`), so `on_close` runs only if that
  body calls `super().close()` (`:853-866,884-899`), after it. Changelog
  entry: a subclass passing both now loses its authored body. Justification:
  MDL-21, one rule for wrapped and subclassed policies.
- **MDL-P7** A subclass's declarations (a derivable `predict`, a static
  `native_chunk` with a chunk corner) are checked before `load` runs. Today
  `MyModel()` loads first (`py/_models/base.py:618`) and refuses after
  (`:705-709,728-740`), so a bad declaration costs a full weight load.
  Justification: the native construction order (Load), no observable change
  besides `load` not running.
- **MDL-P8** A wrapping `Model(policy)` (a policy object, `_policy is not
self`) reads its declarations from the policy, as the native getters do:
  `allow_fusion` is `getattr(policy, "allow_fusion", True)`, read once after
  `load` with `native_chunk` (MDL-P5) and passed to the worker, and the
  wrapper's `allow_fusion`, `native_chunk` and `workflow_edition` attributes
  report the policy's (`workflow_edition` as COR-P6 reads it,
  `type(policy).workflow_edition`). A wrapped bare callable keeps the
  defaults (`True`, `None`, `None`). Today the worker gets the wrapper's
  `allow_fusion=self.allow_fusion` (`py/_models/base.py:507,1073`), the class
  default `True`, so a policy declaring `allow_fusion = False` is fused
  anyway, and the wrapper's `native_chunk` and `workflow_edition` attributes
  read `None` although chunking already follows the policy's K
  (`:742-750`; the policy is kept as `inst`, `py/_models/_coerce.py:66-77`).
  Changelog entry: a wrapped policy that sets `allow_fusion = False` is no
  longer fused. Justification: D-16; a policy's declarations are the model's
  in every form, as a subclass's own class attributes are and as the native
  getters read the policy type's (Authoring), so the wrapper class's
  defaults are an accident. The wrapped object's `reset` hook and its
  no-`load` construction stay as they are (Authoring, MDL-10).
- **MDL-P9** Python's local chunk split follows the native `split_chunk`
  rule (Dispatch by route kind): it splits a `list` into its items, an array
  or tensor along its leading axis (a 0-d one is one frame) and a mapping
  per field; a tuple, a string or bytes, a scalar and any other value are
  one frame. Today `_chunk.split_chunk` calls `list(raw_action)` on every
  iterable that is not a string, bytes or mapping
  (`py/_models/_chunk.py:37-56`), so a returned tuple is split into frames,
  while the native rule keeps `Value::Tuple` whole
  (`crates/rlmesh-adapters/src/stateful.rs:379-423`) and the same model
  replays differently in a local session and when served. The bridges'
  de-chunk helpers follow the same rule: `_first_frame` returns a tuple
  whole and `_chunk_len` measures a tuple as one frame
  (`py/_models/base.py:206-232,235-250` treat a tuple as the frame axis
  today, although `_first_frame` documents byte-consistency with the native
  `split_chunk`), so de-chunk at horizon 1, local replay and native
  `split_chunk` agree. Changelog entry: a model returning a tuple as a chunk
  now replays (and de-chunks) it as one frame; return a list of frames
  instead. Justification: D-16; one chunk rule for local and served replay
  and for de-chunking.
- **MDL-P10** `Model.run` (the in-process native engine) raises `ValueError`
  for a chunk that is not exactly the declared `native_chunk` frames and for
  an empty chunk at a horizon above 1, as a local Session already does
  (`py/_models/_chunk.py:123-129,144-147`). Today the engine raises
  `Error::Model` for both (`rs/model/engine.rs:614-622,727`), which Python
  surfaces as `RuntimeError("model error: …")` (`pyrs/types.rs:50`), and as
  `ModelException` after P-1. RUN-P1 covers only the horizon checks before
  dialing. Changelog entry: these chunk failures in `Model.run` are now
  `ValueError`. Justification: D-16; one class per refusal on both loops
  (Fallback rules).
- **MDL-P11** `Model.run` and `serve` warn `RuntimeWarning` through R-23's
  handler when an undeclared-K model returns a chunk shorter than the
  horizon, at most once per episode, with the Session's text. Today the
  engine only logs `tracing::warn!` once per endpoint
  (`rs/model/engine.rs:600-640`), so neither emits a Python warning, while a
  local Session warns once per episode (`py/_models/_chunk.py:130-143`,
  `py/_models/_eval.py:820,930,1067`). Changelog entry: `Model.run` and
  `serve` may now emit this `RuntimeWarning`. Justification: R-23, D-13; one
  once-scope on both loops (Fallback rules).

Model failures raise `ModelException` per P-1 ([rules](rules.md#python-changes)); the capacity refusal
stays builtin `RuntimeError`.

## Current differences

### Rust

- Authoring is `ModelHandler` / `ModelWorker` / `PredictFn`
  (`rs/model/handler.rs:131`, `rs/model/worker.rs:20-26`,
  `rs/model/predict_fn.rs:28`), not `Policy` + `Model` (G-6). Add `Policy`,
  `Model`, `ModelOptions`, `Corners`, `with_context`, `SpecArg`, `NO_ADAPTER`,
  `rlmesh::types::{PredictContext, BatchPredictContext, EpisodeState}`; move
  the old types to `binding` (R-25) and add `binding::Model::try_new` with
  runtime corner declarations for the Python and C bindings.
- `run_local(self)` / `serve(self)` consume the worker
  (`rs/model/worker.rs:275,334`) and a run fires `on_close`
  (`:72,109,317-323`; G-2, X-7). Target: `&self`, close only on `close()`,
  last Drop, or serve stop.
- `PredictFn::allow_fusion` defaults to `false`
  (`rs/model/predict_fn.rs:141-143`); target `true`.
- `EpisodeInfo { episode_id, seed }` (`rs/model/types.rs:12-18`) has no
  `predict_index`, `predict_seed` or `state`. Move the store (today in
  `py/_models/_episodes.py` and `capi/model.rs:84-160`) into the facade
  `Model`.
- The `PredictFn` seam carries no `env_id` (`rs/model/predict_fn.rs:37,82,165`)
  and `reset_adapter` returns on empty ids (`rs/model/engine.rs:1209-1221`).
  Add `env_id` per predict and `on_route_reset(env_id)` /
  `on_route_release(env_id)`; not a wire change.
- No `load`, `from_config`, `describe` or `spec` on a model; `native_chunk`
  and `allow_fusion` exist only on the `PredictFn` seam
  (`rs/model/predict_fn.rs:141-156`), and `workflow_edition` only on
  `RunLocalOptions` (`rs/model/worker.rs:93`) and serve options.
- `ServeOptions::token` and `ServeModelOptions::token` (two fields, nested one
  wins, `rs/model/worker.rs:212-246,380-391`) and
  `ServeOptions::predict_concurrency` (`rs/serve_options.rs:35-55`, `0`
  clamped to 1 at `rs/model/server.rs:279-281`) → one
  `ext::ModelServeOptions { token, predict_concurrency }` behind
  `ModelServeOptions::ext` (MDL-20), with `Some(0)` refused.
  `RLMESH_MODEL_ENDPOINT_TOKEN` keeps winning (`rs/serve_options.rs:144-170`).
  Every token failure is `Error::Server` today (`rs/serve_options.rs:149-163`):
  the variable's failures become `Error::Internal` and a configured `ext.token`
  that cannot travel in an HTTP header becomes `Error::InvalidArgument`
  (Serve, as Env's step 8).
  `serve` takes no cancel today; add `ModelServeOptions::cancel`.
- `RouteConfig::new` takes `customs: Box<dyn CustomTransform>`
  (`rs/model/predict_fn.rs:204-209`, trait at
  `crates/rlmesh-adapters/src/apply.rs:31-41`), but the facade has no
  entrypoint registry, so routes pass `NoCustoms` (`apply.rs:44-58`) and a
  trusted custom input fails late at apply. Target: MDL-11.
- The engine runs corners and hooks through `spawn_blocking`
  (`rs/model/engine.rs:1125-1251`), so a policy's sync rlmesh call there is
  refused by R-16's async-context probe. Target: the Model's call queue runs
  them on plain library threads with the driving flag clear (R-16
  "Library-run callbacks"; Lifetime).
- The engine's chunk refusals are `Error::Model`: `take_prefix`'s
  wrong-length chunk (`rs/model/engine.rs:614-622`) and the empty chunk
  (`:727`) become `Error::InvalidArgument` (Fallback rules, MDL-P10), and
  `check_horizon_doors`'s "declares native_chunk but defines no chunk
  corner" (`:1283-1291`) moves to construction, after `load`, as
  `Error::Type` (Load, MDL-P5).
- An undeclared-K short chunk warns once per endpoint through
  `tracing::warn!`, behind an endpoint-wide `AtomicBool`
  (`rs/model/engine.rs:600-640`). Target: R-23's `RuntimeWarning` through
  the warning handler, with Python's text, once per episode (Fallback rules,
  MDL-P11).
- `ModelWorker::serve` passes `None` drain and close timeouts through as
  "wait indefinitely" (`rs/model/server.rs:96`, `rs/serve_options.rs:31-34`);
  only Python's binding applies the 5 s cap (`pyrs/model.rs:84,888-891`).
  Target: the facade's `Model::serve` applies ENV-7's caps (Serve).
- `bind_async` / `BoundModelServer` (`rs/model/worker.rs:360`),
  `run_local_cancellable_async`, `run_local_hooked_async` (`:294,310`) move to
  `binding` until replaced (R-25).

### C

- One batched callback `int predict(user_data, obs, out_actions[num_envs])`
  plus `void on_episode_end(user_data, env_id, episode_id /* NULL = all */)`
  and `void on_close` (`h:431-436`). Target: the vtable above.
- `RLMeshEpisode { id, seeded, seed, predict_index, predict_seed }`
  (`h:381-387`), `predict_seed` 0 when unseeded. Target:
  `RLMeshPredictContext` with Python names, `has_*` fields and `struct_size`.
- The store evicts the least-recently-used episode at 4,096 and fires
  `on_episode_end` for it (`capi/model.rs:73,140-160`; G-3). Target: never
  evict; refuse past 65,536 with `RLMESH_ERR_RUNTIME`; entries only for
  `context_corners`.
- `rlmesh_model_new(vtable, user_data, out)` (`h:444`) has no options
  (X-6). Missing: `rlmesh_model_close`, `_clone`, `_from_config`,
  `_describe`, `rlmesh_model_vtable_describe`, `_spec`, `_allow_fusion`, `_native_chunk`,
  `_workflow_edition`, `rlmesh_predict_seed`, `rlmesh_episode_state_*`.
- `CModelHandler` implements `ModelHandler` directly (`capi/model.rs:254,310`);
  a spec reaches C only as JSON through `rlmesh_adapter_resolve` (`h:344`), and
  a C model never fuses (default per-observation `predict_grouped`,
  `rs/model/handler.rs:196-205`). Target: route through the facade `Model`.
- `rlmesh_model_cancel` is terminal and the one allowed re-entry
  (`h:502-512`); a cancelled serve returns `RLMESH_OK` after `on_close`.
  Target: removed; R-18's `RLMeshExtCancelToken` in
  `RLMeshModelServeOptions.cancel` (and Running's options), and a cancelled
  serve returns `RLMESH_ERR_CANCELLED` after `close` (X-8).
- `rlmesh_model_serve(model, bind_address, const RLMeshServeOptions*)`
  (`h:485-496`) carries `token` and `predict_concurrency` inline. Target:
  `const RLMeshModelServeOptions*`, with both in `RLMeshExtModelServeOptions`
  (MDL-20). Its `drain_timeout_ms` / `close_timeout_ms` of 0 ("unset") reach
  the server as `None`, waiting indefinitely (`h:482-493`,
  `capi/model.rs:823`). Target: an unset timeout in Env's
  `RLMeshServeOptions` takes ENV-7's 5 s cap, applied by the facade (Serve).
- `rlmesh_model_free` (`h:517`) does not close first. Target: the last
  reference closes.
- C callbacks are dispatched through `spawn_blocking`
  (`capi/model.rs:365-494`). Target: plain library threads with the driving
  flag clear, through the facade `Model`'s call queue (R-16 "Library-run
  callbacks"), so a C callback may make sync rlmesh calls.

### C++

- `Model::from_predict` / `from_predict_batch` return `Result<Model>` and
  `from_predict` refuses `num_envs != 1` (`hpp:957-972`). Target: the
  templated constructor detecting members, throwing (D-5), Python's spec-less
  dispatch.
- `Episode`, `Batch`, `Request` (`hpp:811,820,866`). Target:
  `types::PredictContext` and per-corner parameters.
- Hooks are chained setters `on_episode_end(EpisodeEndFn(env_id, episode_id))`
  and `on_close` (`hpp:953,975-983`) whose failures have nowhere to go
  (`hpp:1114-1135`). Target: `ModelOptions` fields; throws propagate (R-12).
- `Model::cancel()` (`hpp:1053`) and `Status serve(bind_address, …)`
  (`hpp:1034`). Target: `ModelServeOptions::cancel` (R-18) and
  `void serve(address, const ModelServeOptions& = {})`.

### C#

- Design record (D-45): v0.2 builds no C# model surface. The C# column
  above records the mapping over the C vtable and functions one to one
  (D-17) for v2's binding; it is not a backlog.
- `Model.ServeAsync` would need X-15's `no_signal_handlers` (MDL-24); X-15
  ships in v0.2 for C and C++.
- The describe type form `Model.Describe<TPolicy>(declarations)` cannot
  read a type's declarations without an instance (netstandard2.1 has no
  static interface members, R-28 forbids member reflection), so the caller
  passes the same `ModelDeclarations` the type returns, and passing `null` for a type that implements `IModelDeclarations` is refused rather than silently describing no declarations (MDL-23, COR-28).

## Decisions

| ID     | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Status                  |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| MDL-1  | Concrete `Model` plus a `Policy` trait / vtable / detected class, at the crate root ([rules](rules.md#shapes)). Context types live in `rlmesh::types`, mirroring `rlmesh.types`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Settled                 |
| MDL-2  | `Model::new(policy, ModelOptions) -> Result<Model>` loads eagerly, as `MyModel()` does (`base.py:612-618`). No builder on the handle (R-8), so the resolved spec getter is plain `spec()`; the proposed `get_<name>` rule is dropped.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Settled                 |
| MDL-3  | Rust declares corners with `const CORNERS` / `const CONTEXT` and `<corner>_with_context` twins, because Rust has no overloading or override detection. No proc macro; one may be added later as pure sugar (R-12).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Settled                 |
| MDL-4  | Context opt-in is per corner in every language, as in Python; C uses `context_corners` bits.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Settled                 |
| MDL-5  | Closures: one blanket `impl Policy for F: FnMut(Value) -> Result<Value>`; a closure that wants the context uses `rlmesh::with_context`, the R-3 callable translation (not an R-25 extra).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Settled                 |
| MDL-6  | Raw `Value` never de-chunks (`base.py:391`), in every language, Python included.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Settled                 |
| MDL-7  | Corner and hook calls on one Model are serialized everywhere, Python's local sessions included (MDL-P3), through one FIFO call queue per Model that also takes a `Session` Drop's detached `on_episode_end` (R-17); `close` waits for queued calls as for in-flight ones. The queue runs corners and hooks on plain library threads with the driving flag clear (R-16 "Library-run callbacks"). Costs threaded Python sessions their overlap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | User (accepted default) |
| MDL-8  | Env-tagged entries are dropped silently on evict-all and release, serialized behind in-flight calls, in every language (MDL-P4).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | User (accepted default) |
| MDL-9  | The last reference's Drop / free / destructor calls `close` once if the user did not; errors are discarded (explicit `close()` reports them) and the C++ destructor is `noexcept`. Inside an async runtime a Rust Drop never blocks: it hands `close` to a detached library thread (R-17). Python keeps no close-on-GC (R-3 lifetime row).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Settled                 |
| MDL-10 | A native policy maps to a Python `Model` subclass (Authoring). Python's wrapped-object conveniences for duck-typed third-party objects, the `reset()` episode-end hook (`_coerce.py:66-77`, where the object's `on_episode_end` is not read) and construction without `load()` (`loaders.py:368-381`), are deliberate Python-only behaviour and are not carried over; natively the hook is always `on_episode_end` and `load` always runs once. C++ refuses any policy with a member named `reset` of any signature (`&P::reset`, or `reset()` / `reset(std::string_view)` callable on an overload set), with or without `on_episode_end`, as Python refuses any subclass that defines `reset` (`base.py:867-872`). Rust and C cannot see such a member (an inherent method, no slot), so nothing is refused there. Revised r3 (it refused only `reset` without `on_episode_end`) and r4 (it refused only a nullary `reset`, so the common `reset(episode_id)` compiled and was never called).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Settled                 |
| MDL-11 | `trust_entrypoints` stays Stable natively. Native custom transforms come from a fenced process-wide registry of `ObsTransform`s, for `Custom` inputs only: Rust `rlmesh::ext::register_entrypoint(name: &str, transform: ObsTransform) -> Result<()>`, C `RLMeshStatus rlmesh_ext_register_entrypoint(const char* name, const RLMeshObsTransform* transform)`, which takes its own reference only on `RLMESH_OK` (ADP-7; a replaced name's reference is released), C++ `void rlmesh::ext::register_entrypoint(std::string_view name, adapt::ObsTransform transform)` ([Adapters](adapters.md)). `name` is matched exactly against the spec's entrypoint text; an empty name is the ValueError class. Re-registering a name replaces it for later resolves; routes already resolved keep the transform they hold. Custom-encoding string arms are not looked up: they always fail, as in Python (ADP-11, `python/rlmesh/src/rlmesh/adapters/resolver.py:174-186`). A trusted entrypoint resolves only when registered; an unregistered or untrusted one is refused at resolve with `AdapterResolutionError`, not at apply (ADP-11), the class Python's failed entrypoint import also raises after P-10.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | User (accepted default) |
| MDL-12 | `serve`'s one keyword, `options`, becomes R-6's `ModelServeOptions { options: Option<ServeOptions>, cancel, ext }`, as Env's `EnvServerOptions` carries `options` and `ext` (ENV-12): Env's `ServeOptions` stays the shared struct (`Default` = Python's `None`, `py/_editions.py:251-282`), and the model-only `cancel` (R-18) and fenced extras need a home that is not shared with env servers. Revised r2; it said `serve` takes `ServeOptions` directly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Settled                 |
| MDL-13 | `Model.describe()` reads only static declarations and calls no user code (`base.py:520-525`, `py/_describe.py:190-200`): Rust and C++ describe the type, with no options. C describes the type too, through Core's model type form `rlmesh_model_vtable_describe(&vt, NULL, &out)` (COR-10): the vtable's static fields only, no slot called (not even `load`), no `user_data`; C++ `Model::describe<P>()` wraps it over the static vtable generated for `P` and constructs nothing. The loaded `native_chunk` is never reported, as COR-10 says; Python's describe reads `native_chunk` from the type, never a K `load()` set (COR-P7). The envelope's `params` lists the declared load keys (`LOAD_KEYS`, `load_keys`; none gives `[]`) and `target.qualname` is `rust:<type_name::<P>()>` or `native:model` (COR-25). `rlmesh_model_describe(model, …)` stays Core's instance form. Revised r3 to name the values, r4 to use Core's type form instead of the handle form, r5 to cite COR-25 and COR-P7.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Settled                 |
| MDL-14 | `from_config` takes only the load config; spec and hooks come from the policy, as from Python's class (`base.py:527-551`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Settled                 |
| MDL-15 | `rlmesh_episode_state_set` accepts a NULL value, as Python's dict holds `None`; `_get` returns `bool` with an out-parameter. State keys are author-chosen names, NUL-terminated. `_set` follows R-20: `value` and `free_value` pass to the state only on `RLMESH_OK`; on failure the caller still owns `value`. Revised r4; it took ownership even on failure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Settled                 |
| MDL-16 | Episode entries are created eagerly for context-reading corners, not lazily on first state access: `predict_index` must advance on every such predict.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Settled                 |
| MDL-17 | The capacity refusal is the builtin `RuntimeError` class (R-21 "local refusals"), not `Error::Model`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Settled                 |
| MDL-18 | C and C++ read `allow_fusion` statically from the vtable / class; only `native_chunk` has a post-load slot (`loaded_native_chunk`), because checkpoints decide K. C++ cannot name a static member and a function alike, so the post-load function is `loaded_native_chunk()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Settled                 |
| MDL-19 | `load`'s accepted keys are a static declaration (`LOAD_KEYS`, `load_keys`) the library checks before `load`, in Python's order (missing required first, then unknown keys, sorted), with the ValueError class and the texts of Python's no-`params` path: `MissingParamError`'s, and `UnknownParamError`'s passthrough text with `load` in place of Python's `<Cls>.load` qualname (Load); None forwards every key, as `**kwargs` does. Natives have no annotation kind check: a `load` refuses a wrong-kind value itself with the ValueError class. `params` (`ParamSpec`) stays deferred (Experimental), so the signature is the only rung.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Settled                 |
| MDL-20 | Model serve extras are fenced in `ext::ModelServeOptions { token, predict_concurrency }` (C `RLMeshExtModelServeOptions`), the Model counterpart of ENV-12; `peer_info_extra` is not carried, because the model server ignores it (`rs/serve_options.rs:83-88`). `RLMESH_MODEL_ENDPOINT_TOKEN` wins over `ext.token`, as today. Signals follow ENV-11 for sync `serve` (natively a second SIGINT restores the default disposition and re-raises); `serve_async` installs none (ENV-29); a cancelled serve drains, closes and fails with the cancellation class, and a dropped polled `serve_async` leaves the same state (R-18): the server stops and the drain and `close` finish detached on a library thread (as ENV-32), errors discarded. Revised r6; it kept the model open on a dropped future.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Settled (R-25)          |
| MDL-21 | `on_close` and `on_episode_end` options replace the policy's `close` / `on_episode_end` in every language, for wrapped and subclassed policies alike, as the Python docstring states (`base.py:453-455`); Python's subclass `close` is fixed to match (MDL-P6). The policy's `close` does not also run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Settled                 |
| MDL-22 | A failed constructor runs no hook and frees nothing it was given (R-20): declarations and sizes are checked before `load`, and a failure after a successful `load` (the loaded `native_chunk`) still calls neither `close` nor `on_close`, as Python's `MyModel()` does when it raises after `load`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Settled                 |
| MDL-23 | C# model authoring (D-32). `IPolicy` is an empty marker extended by eight corner interfaces, one per corner and form (`IPredict`, `IPredictWithContext`, `IPredictBatch`, `IPredictBatchWithContext`, `IPredictChunk`, `IPredictChunkWithContext`, `IPredictChunkBatch`, `IPredictChunkBatchWithContext`), detected with `is` at construction and mapped to C's slots and `context_corners`; implementing both forms of one corner, no corner at all, or a static `NativeChunk` without a chunk corner is the TypeError class. `executionHorizon` is always passed, as `uint` (Rust, C). Batch corners take and return `IReadOnlyList<Value>` (and contexts as `IReadOnlyList<PredictContext>`): a `ReadOnlySpan<Value>` parameter cannot be captured or passed to LINQ and is awkward for implementers, and the wrapper already materializes one managed `Value` per row. Hooks are one-member interfaces with one form each: `ILoad.Load(Metadata)`, `IOnEpisodeEnd.OnEpisodeEnd(string)`, `IClose.Close()`, `ILoadedNativeChunk.LoadedNativeChunk`. `IClose` is not `IDisposable`: the Model calls it once at model close (Lifetime), while `IDisposable` is the owner's protocol, and a policy that is also `IDisposable` (or a `MonoBehaviour`) is never disposed by the library. Type-level declarations are a `public sealed class ModelDeclarations` with `{ get; set; }` properties initialized to Python's defaults (`Spec`, `NativeChunk`, `LoadKeys`, `AllowFusion = true`, `WorkflowEdition`), D-22's convention, as `EnvFactoryDeclarations` (ENV-36); the library copies them when it reads them, so mutating the object afterwards changes nothing already built. They are returned by `IModelDeclarations.Declarations` (read once, at construction, before `Load`) and passed directly to `Model.Describe<TPolicy>(declarations)`, which refuses `null` for a `TPolicy` implementing `IModelDeclarations` with the TypeError class (COR-28), the counterpart of C's static vtable fields and `rlmesh_model_vtable_describe`; corners for the type form come from `typeof(IX).IsAssignableFrom(typeof(TPolicy))`, a type check that IL2CPP and trimming preserve. `Model.FromConfig(IPolicy policy, Metadata config)` takes a constructed, unloaded policy, as C's `from_config` takes `user_data`: no `new()` constraint and no `Activator`, so a policy that cannot be default-constructed (a `MonoBehaviour`) works. Delegates: `Func<Value, Value>` and `Func<Value, PredictContext, Value>` constructor overloads (MDL-5's closures). | Pending (C#)            |
| MDL-24 | C# `Model.ServeAsync` installs no signal handlers, as Rust's `serve_async` (R-16 serve exception, ENV-29): it runs the sync `rlmesh_model_serve` on a pool worker with `RLMeshExtModelServeOptions.no_signal_handlers` set (X-15) and the twin's per-operation token in the options' `cancel` field (R-18), so the `CancellationToken` stops the serve (drain, close) and the task ends `Canceled`. The serve has one `cancel` field, so a caller who sets `Cancel` and passes a cancellable token is refused with the ValueError class before scheduling until v2 composes tokens natively (R-18's C# bullet); either alone works. `Ext.ModelServeOptions.NoSignalHandlers` (default `false`) lets an engine host that serves from its own thread with the sync `Serve` skip them too. A field rather than a `rlmesh_model_start` handle, because no language has a model server handle to map (X-15). Design record (D-45): v0.2 builds no C# model surface; X-15 ships for C and C++.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Pending (C#)            |
| MDL-25 | The C# `PredictContext` (and its `EpisodeState`) is lent for one corner call, as C's `RLMeshPredictContext` is: the wrapper invalidates it when the call returns, and any later use throws `ObjectDisposedException` (D-41) instead of reading a freed store. State values are `object?` held by a `GCHandle` that `free_value` releases when the entry is replaced, removed or dropped at episode end (on whichever library thread drops it), a root of the model's ownership record (R-20, R-28), so R-28's shutdown barrier accounts for every one; a value type is boxed, so `GetOrAdd<long>` returns a copy and an author stores the new value back (or stores a class).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Pending (C#)            |

### Review notes

- Applied r1-model 1: C context opt-in is per corner (`context_corners`,
  MDL-4), as `base.py:119-139,674-687` wraps each corner on its own.
- Applied r1-model 2: C++ uses `static native_chunk` plus
  `loaded_native_chunk()` (MDL-18).
- Applied r1-model 3: the default `load` accepts only an empty config
  (`loaders.py:388-398`).
- Applied r1-model 4: Rust loads eagerly in a fallible `Model::new` (MDL-2).
- Applied r1-model 5: terminal close, refusal through sessions and model-wide
  serialization are listed as MDL-P1 to MDL-P3.
- Applied r1-model 6: C `native_chunk` uses `has_native_chunk` (R-13).
- Applied r1-model 7: `CustomTransform` exists (`apply.rs:31-41`,
  `predict_fn.rs:204-209`); the missing piece is a facade entrypoint registry
  (MDL-11).
- Applied the r1-model defaults: concrete `Model` + `Policy`, `with_context`,
  per-corner C flags, eager Rust, raw-`Value` de-chunk restriction, serialized
  env-tagged cleanup, close-on-drop with a non-throwing destructor,
  `trust_entrypoints` plus a fenced registry.
- Consistency pass: `workflow_edition` is an associated function, as on
  `EnvFactory` (ENV-14); C `Model.describe()` is Core's handle form (COR-10);
  MDL-11's registry takes Adapters' callable types, and ADP-11 now defers to
  it.
- Superseded round-0 choices: the required `#[rlmesh::policy]` macro and the
  `rlmesh-macros` crate (R-12, MDL-3), compile-time-only refusal and
  `RLMESH_ERR_INVALID_ARGUMENT` for invalid corners (Python raises
  `TypeError`), `Error::Model` for the capacity refusal (MDL-17), and the
  `rlmesh_model_cancel` re-entry exception (R-18).
- Applied r2 (serve extras, two findings): `ext::ModelServeOptions { token,
predict_concurrency }` behind `ModelServeOptions::ext` (MDL-12, MDL-20),
  token precedence stated under Serve.
- Applied r2 (serve cancel and signals): `ModelServeOptions::cancel`, ENV-11
  signals and the cancelled result under Serve (MDL-20).
- Applied r2 (load keys): `LOAD_KEYS` / `load_keys` checked by the library
  before `load` (MDL-19); the Shapes declare them.
- Applied r2 (`on_close` on a subclass): MDL-21, MDL-P6.
- Applied r2 (constructor failure): construction order and ownership under
  Load and after the C declarations (MDL-22, MDL-P7; R-20 now states it).
- Applied r2: Shapes use only Core's `Metadata` / `Value` API; C++
  `allow_fusion` is static (MDL-18); MDL-11 registers `ObsTransform` only and
  defines re-registration; Rust current-difference bullet corrected; MDL-P5
  reworded; ValueError class for chunk refusals; `EpisodeState` wrong-type
  rule; constrained C++ constructor; `rlmesh_model_workflow_edition` returns
  `RLMeshBytes`; citations `_episodes.py:118`, `:100-107`,
  `_describe.py:190-200`.
- Cross-file (r2): env.md — Current differences (Rust `token`,
  `predict_concurrency` bullet; C `RLMeshServeOptions` bullet): the model's
  `token` and `predict_concurrency` go to Model's `ext::ModelServeOptions` /
  `RLMeshExtModelServeOptions` (MDL-20), not Env's `ext`; the heading
  "`ServeOptions` (also taken by `Model.serve`)" becomes "(also
  `ModelServeOptions.options`)".
- Cross-file (r2): env.md — ENV-12: state that `RLMESH_ENV_ENDPOINT_TOKEN`,
  when set, wins over `ext.token` (today's rule, `rs/serve_options.rs:144-170`),
  matching MDL-20, or tell Model the opposite precedence.
- Cross-file (r2): core.md — Values row: list C++ `Value` converting
  constructors (at least `Value(int64_t)`, `Value(double)`, `Value(bool)`,
  `Value(std::string)`) and Rust `Metadata: FromIterator<(String, Value)>`,
  which the Model Shapes use.
- Applied r3 (13 findings, none rejected): Rust default corner bodies return
  `Error::Runtime` (R-21 folds `NotImplementedError`), C and C++ cannot reach
  an undefined corner; the Rust limitation on undetectable unwritten corners
  is stated under Authoring; `LOAD_KEYS` refusals follow Python's no-`params`
  path (missing first, then unknown keys sorted, passthrough text with
  `load`; MDL-19); Serve signals defer to ENV-11 and `serve_async` installs no
  handlers (ENV-29, MDL-20); C++ `cancel` is `std::optional<ext::CancelToken>`;
  a dropped `serve_async` fires no hook and a last-reference Drop inside a
  runtime detaches `close` (MDL-9); endpoint-token validation matches Env
  step 8; construction-order note scoped to `MyModel()`; C++ refuses any
  nullary `reset` (MDL-10); C describe names its values (MDL-13); Serve
  states the handshake edition; the horizon warning covers every row without
  a chunk corner; `impl From<ModelSpec> for SpecArg`; `EpisodeState` /
  `LoadKey` are native translations; MDL-11 cites the full resolver path.
- Applied the rules r3 cross-file notes: C token `RLMeshExtCancelToken`
  (Hooks table, `RLMeshModelServeOptions`, C current differences); MDL-7 /
  Lifetime FIFO queue taking a `Session` Drop's `on_episode_end`, `close`
  waiting for queued calls; MDL-11 cites P-10.
- Cross-file (r3): rules.md — R-22: next to the Rust undefined-corner
  exception, list the native `LOAD_KEYS` unknown-key text
  `extra='passthrough' but load has no **kwargs to forward: a, b`, which
  names `load` where Python names the qualname `<Cls>.load`
  (`params/_resolve.py:153-157,455-456`), because native policies carry no
  type name (MDL-19).
- Applied r4 (11 findings, two duplicates, none rejected): C
  `Model.describe()` is Core's type form `rlmesh_model_vtable_describe`
  (MDL-13, core r3 note); the C registry takes `const RLMeshObsTransform*`
  (MDL-11, adapters r3 note); MDL-P8 reads a wrapped policy's
  `allow_fusion`, `native_chunk` and `workflow_edition`, and the edition row
  cites COR-P6 (core r3 note); `close()` stops a running serve and refuses
  every later corner or hook call, Sessions opened earlier included
  (Lifetime, MDL-P2); C++ refuses a `reset` of any signature (MDL-10);
  full `EpisodeState`, `with_context` / `WithContext` and `Corners`
  signatures; `rlmesh_episode_state_set` follows R-20 (MDL-15); MDL-P3 lists
  Python's re-entry refusal; invalid `context_corners` bits are the TypeError
  class; `_chunk.py:123-129,144-147` and `base.py:618` citations.
- Cross-file (r4): running.md — Session lifetime / RUN-9: after the model's
  `close()`, `predict`, `step` and `run` on a Session over that local `Model`
  fail with the builtin `RuntimeError` class (closed handle) and call no user
  code, and a Session closed or dropped afterwards fires no `on_episode_end`
  (Model Lifetime, MDL-P2).
- Cross-file (r4): rules.md — R-17 "Session Drop" and X-7: a Drop after the
  model's `close()` submits nothing to its queue; `close()` also stops a
  `serve` still running on the model (drain, that `serve` returns success)
  before firing the hook (Model Lifetime).
- Cross-file (r3): core.md — COR-10: the instance-form sentence "`spec` and
  `native_chunk` come from the instance" contradicts its own Model sentence
  and COR-P3 (class-level `native_chunk`); state that for a `Model` only
  `spec` comes from the instance and `native_chunk` is the policy's static
  declaration (MDL-13), or tell Model to report the loaded K.
- Applied r5 (10 findings, none rejected): the endpoint-token classes split
  as Env's step 8 (variable `RLMeshException`, `ext.token` the ValueError
  class), with the Rust `Error::Server` change listed; a native policy maps
  to a Python subclass for hooks, `load` and declarations, and the wrapped
  object's `reset` hook and no-`load` construction are listed as Python-only
  (Authoring, MDL-10, MDL-P8's justification); C++ detects nullary
  `load()` (= `load_keys = {}`) and `on_episode_end()`; `close()`'s drain
  finishes admitted requests, and `close()` / a last-reference Drop inside a
  corner or hook fails or detaches instead of deadlocking (MDL-P3 lists
  `close()`); Rust's const declarations can miss corners in both
  directions; tuple chunks are one frame, Python aligns (MDL-P9); C
  `load_keys` + `load_keys_len` follow R-13; the GC row names no Python
  context manager; the rules r4 notes (R-22 default-`load` text, plain
  library threads with Rust and C current differences, MDL-7) and the core
  r4 note (MDL-13: COR-25 `params` / `qualname`, COR-P7) are applied.
- Cross-file (r5): core.md — COR-10: the C type form reads `load_keys` +
  `load_keys_len` as params (R-13: NULL = None, non-NULL with 0 = `[]`), not
  `has_load_keys` + `load_keys` (Model's vtable dropped `has_load_keys` and
  renamed `load_key_count`).
- Cross-file (r5, optional): rules.md — R-17: a last-reference Drop / free /
  destructor of a `Model` from inside that model's own corner or hook takes
  the detach path (hands `close` to a library thread) rather than waiting
  for its own call (Model Lifetime).
- Applied r6 (10 findings, none rejected): the declared-K chunk refusals are
  the ValueError class on every replay path, with the engine's
  `Error::Model` and Python's `Model.run` listed (Rust current differences,
  MDL-P10); undeclared K replays `min(len, horizon)` and warns
  `RuntimeWarning` once per episode, with the `tracing` difference and
  MDL-P11; a dropped polled `serve_async` stops the server and closes the
  model detached, as a token cancel and ENV-32 do (Serve, MDL-20); a loaded
  K with no chunk corner is the TypeError class after `load` (Authoring,
  Load, MDL-P5, Rust current differences); the 5 s serve-timeout caps are
  listed as Rust and C differences; MDL-P9 covers `_first_frame` and
  `_chunk_len`; `RLMESH_PREDICT_CONTEXT_INIT`; `EpisodeState` takes Core's
  `Metadata` names in Rust and C++; `pub trait Policy: Send + 'static`; C++
  `ModelOptions::on_episode_end` is an `EpisodeEndHook` accepting both
  arities.
- Cross-file (r6, optional): rules.md — R-18 test plan (Serving, next to
  ENV-32): dropping a polled `Model::serve_async` stops the server, does not
  block the dropping task, and leaves the model closed (its serve-stop
  `close` runs detached), as a cancelled model serve does (Model Serve,
  MDL-20).
- Cross-file (r6, optional): running.md — RUN-P1 / Validation order: note
  that the per-predict chunk refusals (wrong length against `native_chunk`,
  empty chunk at a horizon above 1) are the ValueError class in `Model.run`
  too, with the Python change in Model's MDL-P10.
