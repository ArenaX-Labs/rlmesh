# Language parity

Status: Reviewed (Opus rounds 2–6); user decisions settled 2026-10-15; implemented in Python, Rust, C (ABI 5) and C++17 (tested also at C++20; D-5 amended 2026-10-16). C# column: a design record of how C# maps onto the v1 surface, C# review rounds 1 and 2 applied 2026-10-20, merged as a document and not implemented as an API in v0.2 (D-45, scope change 2026-10-19). v0.2 implements the C# foundation and the C ABI additions X-12 to X-16, each pending user approval.

RLMesh is one library in five languages: Python (the reference), Rust (the
`rlmesh` facade crate), C (`crates/rlmesh-capi/include/rlmesh.h`), C++
(`rlmesh.hpp`, wrapping C one to one) and C# (the `RLMesh` package in
`csharp/`, also wrapping C one to one; D-17). C# targets game engines and
simulators; engine integrations such as `integrations/unity/` sit on top of
it and are outside this spec, though their constraints shape it (R-28). If
you know the Python API you know what to expect in every other language,
adjusted only by the fixed translation rules. A difference that no rule
explains is a bug.

Python's Stable surface
(`python/rlmesh/tests/api_surface/snapshots/api_surface.json`, labels in
[compatibility](../compatibility.md)) defines the nouns, verbs, defaults and
field names. The wire contract (`crates/rlmesh-proto/proto/`) is frozen and
this spec does not change it.

## C# and the v2 protocol

The RLMesh protocol is being redesigned for v2: clocked sessions, typed
channels, timed action chunks, leases and an edge runtime, on a Zenoh data
plane with a Rust core, every language a thin binding over the core's C ABI.
v0.2 ships on the current wire; v2 is a clean break after it. So:

- **The C# column is a design record.** It records how C# maps onto the v1
  surface, and why, for the v2 bindings to learn from. It is not a backlog:
  v0.2 does not implement the v1-shaped public C# API (the env, factory,
  model and session surface with `Env.Reset` / `Step`, `Model.Predict` and
  `Session.Step`, runs, remote clients, adapters, spaces-first authoring),
  and no part of it ships behind a flag (D-45). The few env-serving types
  v0.2 exposes are Experimental and may change for v2.
- **v0.2 implements** (D-45): the additive C ABI changes X-12 to X-16
  ([below](#c-abi-additions-for-v02)), each pending user approval, and the
  C# foundation: native library loading and the ABI gate, the generated
  interop layer, ownership records and `SafeHandle` types, callback
  trampolines with exception containment (X-11), cancellation plumbing,
  the reload-safe shutdown barrier (D-43), NuGet and UPM packaging
  (`netstandard2.1` + `net10.0`), CI with a NativeAOT stand-in and the test
  harness. Its public surface is only what serving an authored env to a
  Python client needs, labelled Experimental (rules R-28 "v0.2 surface").
- **Enforcement** follows: the manifest's `csharp` column (C# surface
  parity) is deferred to v2; the foundation checks run in v0.2 (rules
  [C# enforcement](language-parity/rules.md#c-enforcement)).
- [Lessons for the v2 C ABI](#lessons-for-the-v2-c-abi) states what the
  core's C ABI must design for from day one.

## How to read

1. Start with [rules](language-parity/rules.md): the translation table, the
   options-struct and async rules, C struct sizing, the one error matrix,
   warnings, fencing, decisions, Python changes, shapes, known gaps and the
   enforcement. C# rules are clauses inside each rule, plus R-28 (C# runtime
   constraints), X-11 (C# trampolines) and X-12 to X-16 (the C ABI
   additions, detailed [below](#c-abi-additions-for-v02)).
2. Then read the area you work on. Each area maps every Stable Python symbol
   it owns to Rust, C, C++ and C#, and lists its own Python changes and current
   differences. A symbol shared by several areas has one owner (rules R-27);
   the others link to it. Area C# cells are design record (D-45).
3. IDs are stable: `R-n` rules, `D-n` decisions, `P-n` Python changes, `X-n`
   cross-cutting changes, `G-n` gaps, `E-n` enforcement. Area files use
   `<AREA>-n` for decisions and `<AREA>-P<n>` for Python changes (`COR`, `SPC`,
   `ENV`, `MDL`, `RUN`, `RMT`, `ADP`). Cite them in code review and commit
   messages.
4. "User (accepted default)" marks a default the user accepted on
   2026-10-15. Every one is listed under
   [Decisions accepted by the user](#decisions-accepted-by-the-user).
   "Pending (C#)" marks a C# default awaiting review; the ones v0.2 needs,
   and each pending C ABI addition, are listed under
   [C# decisions pending user review](#c-decisions-pending-user-review).

## Files

| File                                       | Covers                                                                                                                            |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| [rules.md](language-parity/rules.md)       | Rules, decisions, error matrix, warnings, Python changes, shapes, known gaps, enforcement                                         |
| [core.md](language-parity/core.md)         | `Value`, `Metadata`, error detail fields, workflow editions, describe, build info                                                 |
| [spaces.md](language-parity/spaces.md)     | Spaces, `SpaceSpec`, space values and `Tensor`                                                                                    |
| [env.md](language-parity/env.md)           | Env authoring and serving: `Env`, `VectorEnv`, `EnvFactory`, `Variant`, `EnvServer`, `ServeOptions`, `EnvContract`, `trial_index` |
| [model.md](language-parity/model.md)       | Model authoring: `Policy`, `Model`, `PredictContext`, `SpecArg`, `NO_ADAPTER`                                                     |
| [running.md](language-parity/running.md)   | `run`, `session`, `Session`, run and env targets, `RunContext`, `RunHooks`, results, `View`, `Recorder`, `RANDOM_SAMPLE`          |
| [remote.md](language-parity/remote.md)     | `RemoteEnv`, `RemoteVectorEnv`, `RemoteModel`, their options, sandbox clients                                                     |
| [adapters.md](language-parity/adapters.md) | `EnvTags`, `ModelSpec`, `ObservationRoles`, transforms, `tag`, `resolve`, `Adapter`                                               |

## Decisions accepted by the user

On 2026-10-15 the user accepted every default below as implemented. They are
settled; reversing one is a spec change.

| ID     | Question                                                                                                                  | Default                                    |
| ------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| D-9    | Is `sanitize_metadata` relabelled Bridge, with no native counterpart?                                                     | Yes (P-7)                                  |
| COR-2  | Does `Metadata` keep insertion order like a Python `dict`, rather than sorting keys?                                      | Insertion order                            |
| COR-4  | Is `is_recoverable` false for every class but Environment and Model (no transport retry flag)?                            | Yes                                        |
| COR-14 | Keep `kind` on native describe as a cross-check, and make Python refuse a contradicting `kind`?                           | Yes (COR-P1)                               |
| SPC-6  | Does `Box.contains` check the shape exactly?                                                                              | Yes (SPC-P7)                               |
| SPC-7  | Are non-integral bounds on integer or bool Box dtypes refused instead of truncated?                                       | Refused (SPC-P6)                           |
| SPC-8  | Does a typed-uniform Box read `low` / `high` back as one number?                                                          | Yes (SPC-P5)                               |
| SPC-18 | Keep identity equality for `SpaceSpec` and `Tensor` (no native `==` except Rust `Tensor`), deferring structural `__eq__`? | Yes                                        |
| SPC-21 | Does `RANDOM_SAMPLE` seed its sampler with each episode's reset seed?                                                     | Yes (SPC-P8)                               |
| SPC-22 | Defer per-element Box construction to fenced `ext` builders?                                                              | Yes                                        |
| SPC-23 | Keep Experimental space constructors, deferring their attribute accessors?                                                | Yes                                        |
| ENV-13 | Does only `EnvFactory.serve` publish the describe envelope, not a bare `EnvServer`?                                       | Yes; C keeps it via `ext.publish_describe` |
| ENV-15 | Do native factories declare `TagParam { name, default_, choices }` for the branch table?                                  | Yes                                        |
| ENV-23 | Serve `!Send` Rust envs through a fenced foreground-only `ext::LocalEnvServer`?                                           | Yes                                        |
| MDL-7  | Serialize corner and hook calls on one Model across all sessions, Python's local sessions included?                       | Yes (MDL-P3)                               |
| MDL-8  | Drop env-tagged episode entries silently on evict-all and release?                                                        | Yes (MDL-P4)                               |
| MDL-11 | Keep `trust_entrypoints` natively, backed by a fenced entrypoint registry?                                                | Yes (ADP-11)                               |
| RUN-4  | Keep `Model.run` without a viewer (a set `view` with a local `Model` is TypeError)?                                       | Yes                                        |
| RUN-17 | Promote `FrameSources` and the recorder records to Stable in Python?                                                      | Yes (RUN-P5)                               |
| RMT-2  | Keep the client `address` in the options struct rather than adding it to R-7 as a positional?                             | Options field                              |
| RMT-4  | Does a zero client timeout mean "no timeout", as Python reads it?                                                         | Yes                                        |
| RMT-5  | Keep endpoint tokens off the shared surface, in fenced `ext` constructors?                                                | Yes; superseded by RMT-22, removed         |
| RMT-22 | Share `api_key` and `tls_ca_file` client options and dial `https://` over TLS in every language?                          | Yes                                        |
| RMT-23 | Let a served model drive a plain local env with no `EnvServer`?                                                           | Yes (local vector env refused)             |
| RMT-13 | Does I/O on a closed remote client raise `RuntimeError` instead of `ConnectionError`?                                     | `RuntimeError` (RMT-P2)                    |
| ADP-15 | Give unexported Literal types native names derived from their annotation?                                                 | Yes                                        |
| ADP-16 | Relabel `Adapter.__init__` / `serve_route` Internal and the `*_value` transforms Bridge?                                  | Yes (ADP-P4)                               |
| ADP-17 | Keep the core resolution code in `binding`, exposing only the message on `AdapterResolutionError`?                        | Yes                                        |

## C# decisions pending user review

D-17 (C# wrapping C, for game engines) is the user's, amended by the
2026-10-19 scope change (D-45). Every other C# choice is a recommended
default, written into the rules and area tables as "Pending (C#)", revised
after C# review rounds 1 and 2 (rules' Review notes). The X-12 to X-16 rows
are additive C ABI changes, detailed in
[C ABI additions for v0.2](#c-abi-additions-for-v02), each pending user
approval; D-44's test mechanism is a pending user decision, not a
recommendation. Accepting one makes it settled; rejecting one is a spec
change in the cited rule or area.

### Needed for v0.2

These shape what v0.2 builds: the foundation, its minimal Experimental
surface, its enforcement and the C ABI additions. A row about public types
applies in v0.2 only to the types R-28's v0.2 surface ships (`IEnv`, `Env`,
`EnvServer`, `Space`, `Value`, `Metadata`, `Tensor`, the exception classes,
`Ext.Lifetime`); the rest of such a row is design record.

| ID     | Question                                                            | Recommended default                                                                                                                                                                                                                                                                                                                                                                                                 | Rationale                                                                                                                                                                                                   |
| ------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-18   | How is the C# library laid out and generated?                       | `csharp/` at the root, package and namespace `RLMesh`; raw layer generated by ClangSharpPInvokeGenerator 21.1.8.4 (the spike's flags, `internal`), with a `--check` staleness gate in `mise run check`; hand-written public layer                                                                                                                                                                                   | The spike generated 4.5k lines with no `delegate*` and bound ABI 5 with no changes; a generator keeps the 510 imports in step with `rlmesh.h`                                                               |
| D-19   | Which targets and language level?                                   | `netstandard2.1` + `net10.0`; public surface consumable from C# 9; 64-bit only; tests on the .NET 10 runtime; `#if` only for the net10.0 resolver, the internal net10.0 unload hook (D-43) and `System.Half`; E-2 diffs both assemblies against an allowlist (`System.Half` only); SDK pinned by `global.json`, installed by mise                                                                                   | .NET 8 support ends 2026-11-10; Unity, Mono and Godot need netstandard2.1 and C# 9; the generated layout is per-architecture                                                                                |
| D-20   | How are names cased?                                                | One transform: `PascalCase` members and constants, `camelCase` parameters, `@` for keyword parameters, no trailing `_`                                                                                                                                                                                                                                                                                              | .NET conventions while E-2 stays one mapping per name; `@params` keeps the metadata name equal to Python's                                                                                                  |
| D-22   | Options classes or C# optional parameters for keywords?             | Sealed classes with `{ get; set; }` properties initialized to Python's defaults, `options = null` last; not records, not `init`-only                                                                                                                                                                                                                                                                                | Optional parameters bake constant defaults into callers; records add unreviewed equality; `init` blocks conditional assembly and C# 9 has no `required`. Challenges the `init` starting point               |
| D-24   | Async twins and their workers?                                      | v0.2 has only `EnvServer.ServeAsync` / `WaitAsync` / `ShutdownAsync` (D-45): `async` methods over the sync C call on a library worker pool; the design record's per-handle workers and explicit per-call cancel (X-12) for the rest; sync calls never refused in async code                                                                                                                                         | Engines must not block a frame; `await` rethrows the same `CancelledError`; no ambient cancel state                                                                                                         |
| D-25   | Which .NET bases do the exceptions use?                             | `ValueError : ArgumentException`, `TypeError : InvalidCastException`, `RuntimeError : InvalidOperationException`, `OSError : IOException`, `Ext.CancelledError : OperationCanceledException`; documented: every rlmesh failure class is an `InvalidOperationException`                                                                                                                                              | Keeps .NET idioms while copying Python's tree; `catch (InvalidOperationException)` reads as `except RuntimeError`                                                                                           |
| D-29   | Tuples and Gymnasium results?                                       | Hand-written `readonly struct ResetResult` / `StepResult` (and vector forms) with named properties and `Deconstruct`; other tuples named `ValueTuple`s                                                                                                                                                                                                                                                              | `var (obs, reward, …) = s.Step(a)` reads like Gymnasium and `result.Reward` stays available                                                                                                                 |
| D-31   | Tensor memory?                                                      | No span over native memory on the shared surface: `Buffer()` returns an owning `TensorLease<byte>` (own native reference, no finalizer); typed leases and fenced `DangerousAsSpan<T>` in `RLMesh.Ext`; inputs copied                                                                                                                                                                                                | A span does not keep its tensor alive: no use-after-free on the safe path, a forgotten lease leaks instead                                                                                                  |
| D-32   | Authoring shape?                                                    | Required interface per protocol, one-member optional interfaces detected with `is`, declarations objects (MDL-23, ENV-36), no base class required                                                                                                                                                                                                                                                                   | A `MonoBehaviour` or Godot `Node` can implement it; a default interface method cannot express C's "not defined" and IL2CPP support for it is unestablished; no static abstracts in netstandard2.1; AOT-safe |
| D-33   | Fence?                                                              | `RLMesh.Ext` namespace; raw layer `internal`, no public binding module, no `InternalsVisibleTo` for integrations                                                                                                                                                                                                                                                                                                    | Mirrors C++ `rlmesh::ext`; the Unity layer proves the public surface is enough                                                                                                                              |
| D-34   | Integer types?                                                      | C's widths (`long`, `ulong`, `uint`, `ushort`); `int` for in-memory counts and indices; `[CLSCompliant(false)]`                                                                                                                                                                                                                                                                                                     | One type per E-2 comparison; BCL collections count in `int`                                                                                                                                                 |
| D-35   | Durations?                                                          | `TimeSpan`, `_seconds` dropped, 100 ns precision; `Timeout.InfiniteTimeSpan` is negative, not infinite                                                                                                                                                                                                                                                                                                              | Idiomatic; R-14's checks stay in C                                                                                                                                                                          |
| D-36   | A string with U+0000 bound for a NUL-terminated C input?            | Refused with the ValueError class before the call (unpaired surrogates too); C++ makes the same refusal in the same release; exception text reported to C is sanitized, not refused (X-11)                                                                                                                                                                                                                          | Silent truncation (C++ today) hides bugs; the two wrappers over C agree                                                                                                                                     |
| D-38   | Callback threading?                                                 | Where C runs them: the calling thread for local sync runs and sessions and for factory callbacks, library threads (several at once) for served envs and models; never marshalled                                                                                                                                                                                                                                    | An integration must know when to call inline; queuing an inline call to the main thread deadlocks                                                                                                           |
| D-39   | Finalizers and `Dispose()`?                                         | One ownership record per native handle with a CAS state word (R-28); `Dispose()` closes the owner at once and frees inline only when no reference remains and the thread is not in a callback or a library thread, otherwise the last release queues the free to the cleanup thread; `Ext.Lifetime.WaitFreed` waits for it; finalizer-reached callback-bearing frees always queue                                   | The finalizer thread must not block or reenter authored code; a callback that drops the last reference must not wait on itself; dependencies outlive the native free                                        |
| D-40   | Distribution?                                                       | NuGet with `runtimes/<rid>/native` (64-bit; no WebGL); Unity consumes the prebuilt netstandard2.1 DLL as a UPM package (plus an `__Internal` build for iOS); host runtime `{"language":"csharp","framework_versions":{…}}` through `HostRuntimeJson`, set by the Unity integration                                                                                                                                  | Frees the implementation from Unity's compiler version; C refuses a `toolchain` other than gcc, clang or msvc; detection cannot tell IL2CPP from Mono                                                       |
| D-41   | A call on a disposed wrapper?                                       | `ObjectDisposedException`                                                                                                                                                                                                                                                                                                                                                                                           | A C#-only state, idiomatic, and an `InvalidOperationException` like `RuntimeError`                                                                                                                          |
| D-42   | Enforcement in v0.2?                                                | The foundation checks only: interop staleness, ABI gate, both assemblies built and their public surfaces diffed, IL and reference bans, field-wise layout and `_INIT` checks, trampoline, cancellation-race and shutdown fixtures on the .NET 10 JIT against each assembly, NativeAOT as a supplement, an advisory benchmark; the manifest `csharp` column (C# surface parity) is deferred to v2                    | There is no v1-shaped surface to compare; NativeAOT is not evidence for Unity (D-44)                                                                                                                        |
| D-43   | How does C# survive a Unity domain reload?                          | `Ext.Lifetime.Shutdown(timeout)`: a terminal admission gate, refusal from callbacks and library threads, every blocking teardown on tracked threads under one absolute deadline, completion proven by joined native frees rather than an empty registry; `false` retains everything and `Pending()` says what; the host runs it to `true` before it starts a reload (preflight), unload events are diagnostics only | The caller never blocks past the timeout, and an unload event cannot refuse an unload                                                                                                                       |
| D-44   | When may v0.2 claim Unity support, and how are the Unity tests run? | Gate: an IL2CPP player smoke test, a Mono Editor domain-reload test and a failed-barrier reload test (the reload stays deferred, the Editor stays responsive) pass before v0.2 claims Unity support. Mechanism (Unity licence, manual run or dispatch-only CI): **pending user decision**                                                                                                                           | NativeAOT and Mono-on-.NET runs miss IL2CPP codegen, UnityLinker stripping and the Editor's reload                                                                                                          |
| D-45   | What does v0.2 ship in C#?                                          | The foundation (native loading and ABI gate, generated interop, ownership records, trampolines with exception containment, cancellation plumbing, the shutdown barrier, NuGet and UPM packaging, CI and test harness) plus a minimal Experimental surface for serving an authored env (R-28 "v0.2 surface"); nothing else of the C# column                                                                          | The user's 2026-10-19 scope change: v2 supersedes the v1-shaped API                                                                                                                                         |
| COR-29 | How is a C# `Value` built?                                          | An exact implicit conversion per type: `bool`, `sbyte`, `byte`, `short`, `ushort`, `int`, `uint`, `long`, `float`, `double`, `string`, `Tensor`, `Metadata`; `char` and `ulong` conversions are `[Obsolete(error: true)]`; `Count`/indexer/`TryGet` without a collection interface                                                                                                                                  | Exact operators make every literal and integer type bind one operator (no CS0457); counterpart of C++'s deleted overloads                                                                                   |
| COR-30 | What is C# `Metadata`?                                              | Managed, mutable, insertion-ordered `IDictionary<string, Value>`; order-insensitive `Equals`/`==`; hash on `Count`                                                                                                                                                                                                                                                                                                  | Python `dict` equality, knowingly against .NET's advice for mutable types                                                                                                                                   |
| SPC-31 | How is tensor memory read and written?                              | Constructors copy; `Buffer()` is an owning `TensorLease<byte>` (v0.2); typed leases, raw spans and `Ext.DlpackTensor` with its import state machine are design record                                                                                                                                                                                                                                               | R-28 lifetime safety; matches Rust's `TensorExt` and C++'s `ext`                                                                                                                                            |
| SPC-33 | Space class hierarchy?                                              | Unsealed `Space` with sealed `Box`/`Discrete`/`Dict`/Experimental kinds; `AsBox()`-style downcasts; cached child wrappers; `Spaces.Tuple` keeps its name                                                                                                                                                                                                                                                            | `is Box` works as Python's `isinstance`                                                                                                                                                                     |
| ENV-36 | How are envs and factories authored?                                | `IEnv`/`IVectorEnv` plus `IRender`, `IClose`, `IRenderMode`, `IMetadata`, `IId`; `IEnvFactory` plus `IPrepare`, `ITagsFor`, `IEnumerateVariants` and `EnvFactoryDeclarations`; public `Env`/`VectorEnv`/`EnvFactory` handles                                                                                                                                                                                        | R-12's pattern; no base class; describe by handle as C does                                                                                                                                                 |
| ENV-37 | One native handle per authored env?                                 | `Env.For` / `VectorEnv.For` only; a registry whose lock never spans native work; R-28's compare-and-swap ownership state machine (`Idle`, `InUse(n)`, `Transferring`, `Transferred`, `Closing(n)`, `Freeing`, `Freed`) shared by calls, borrowing, disposal and factory transfer; `For` adopts the caller-owned record whatever the GC has done                                                                     | No second native handle, no use after transfer, no answer that depends on GC timing                                                                                                                         |
| ENV-38 | How does a served C# env report its runtime?                        | `HostRuntimeJson` on `Ext.EnvServerOptions` and (X-14) `Ext.EnvFactoryServeOptions`; `null` = `{"language":"csharp","framework_versions":{"dotnet" or "mono": version}}`                                                                                                                                                                                                                                            | C accepts no `toolchain` for C#; a prebuilt assembly cannot tell IL2CPP from Mono, so the Unity integration sets it                                                                                         |
| ENV-40 | How does a `CancellationToken` act on server twins?                 | `ServeAsync` checks the token before `Start()` (a bind failure throws before the task is returned); cancel triggers `Shutdown()` on the twin's worker, not in the token callback, and ends `Canceled`; `WaitAsync` waits in 100 ms slices; `ShutdownAsync` cancel stops waiting only                                                                                                                                | C# form of ENV-32's dropped futures; never starts a cancelled server and never blocks inside `Cancel()`                                                                                                     |
| X-12   | C ABI: per-call cancellation?                                       | Seven explicit `rlmesh_ext_` variants taking a `cancel` argument (session reset, predict, step; remote env and remote vector env reset, step), subscribed once at native admission; C++ `ext::` overloads. No thread-local scope. **Pending user approval**                                                                                                                                                         | The v0.2 consumers are those seven calls; calls with an options `cancel` field already have an explicit input; no ambient state                                                                             |
| X-13   | C ABI: a background factory serve?                                  | `rlmesh_ext_env_factory_start(factory, address, options, &server)`: per-call gate collection moved into the server, factory admission released at return, the caller's `cancel` watched for the whole serve; C++ `ext::start`. **Pending user approval**                                                                                                                                                            | Engine hosts keep the factory envelope (ENV-13) without signal handlers                                                                                                                                     |
| X-14   | C ABI: a host runtime for factory serves?                           | `RLMeshEnvFactoryServeOptions.ext` → `RLMeshExtEnvFactoryServeOptions.host_runtime_json`. **Pending user approval**                                                                                                                                                                                                                                                                                                 | A C++ or C# factory serve reports language `"c"` today                                                                                                                                                      |
| X-15   | C ABI: model serve without signal handlers?                         | `RLMeshExtModelServeOptions.no_signal_handlers` (zero = install), not `rlmesh_model_start`. **Pending user approval**                                                                                                                                                                                                                                                                                               | No language has a model server handle; the facade already has the switch                                                                                                                                    |
| X-16   | C ABI: unregister a trusted entrypoint?                             | `rlmesh_ext_unregister_entrypoint(name, expected, &removed)`: removes only the registration `expected` made (NULL = any), dropped outside the registry lock; C++ and Rust `ext` counterparts. **Pending user approval**                                                                                                                                                                                             | A plugin or domain that unloads must not keep a function pointer in the registry, nor remove another owner's                                                                                                |

### Design record (no v0.2 decision)

Recommended defaults for the v1-shaped C# surface that v0.2 does not build
(D-45). They need no decision now; v2's C# binding starts from them and from
[Lessons for the v2 C ABI](#lessons-for-the-v2-c-abi).

| ID     | Question                                                                      | Recommended default                                                                                                                                                                                                                                       | Rationale                                                                                                                                         |
| ------ | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-21   | Where do free functions live?                                                 | `RLMesh.RLMeshApi`, `RLMesh.Spaces.SpacesApi`, `RLMesh.Adapters.AdaptersApi` (`<Module>Api`); type-named builders become constructors; classmethods stay static on their type                                                                             | A class cannot share its namespace's name; a bare `Api` collides with user types; `using static RLMesh.RLMeshApi;` reads like Python; mirrors D-6 |
| D-23   | How do R-7's positional calls look?                                           | Optional parameters defaulting to `null` (= Python's default) or Python's constant for a non-optional scalar                                                                                                                                              | `null` bakes in nothing that can change; named arguments read like Python                                                                         |
| D-26   | Equality and hashing?                                                         | `IEquatable<T>`/`Equals`/`==` where Python has `__eq__`; `GetHashCode` always agrees, content-based where Python hashes (floats included), coarse where it does not                                                                                       | .NET requires hash to agree with equality; .NET doubles are reflexive, so Rust's float restriction does not apply                                 |
| D-27   | Collection types?                                                             | Inputs `IReadOnlyList<T>` / spans, copied; `params T[]` for `*args`; outputs `IReadOnlyList<T>`; record list fields `EquatableList<T>`                                                                                                                    | Record equality on a plain list is reference equality, which is not Python's `==`                                                                 |
| D-28   | Frozen dataclasses?                                                           | `sealed record` classes with an explicit `camelCase` constructor for required fields, `init` properties for defaulted ones, `ToString()` = Python's `str`, `With<Name>` instance methods                                                                  | `with` and value equality for free; positional-record parameters would be `PascalCase` and fail E-2                                               |
| D-30   | Python aliases (`InfoDict`, `Kwargs`, `StackSpec`)?                           | Spelled as their target (`Metadata`, `IReadOnlyList<int>`)                                                                                                                                                                                                | .NET has no exported alias; a subclass breaks argument binding                                                                                    |
| D-37   | Warning handler shape?                                                        | One `Action<Warning>?` via `RLMeshApi.SetWarningHandler`, not an `event`                                                                                                                                                                                  | R-23 is one atomic handler; several subscribers make "the handler failed" ambiguous                                                               |
| COR-28 | How does C# describe a type?                                                  | `RLMeshApi.Describe<TPolicy>(ModelDeclarations? declarations = null, …) where TPolicy : IPolicy` (corners from implemented interfaces); a `TPolicy` implementing `IModelDeclarations` with `null` declarations is TypeError; no env-factory type form     | No static interface members or reflection; never silently drops declarations; C also describes a factory only as a handle                         |
| SPC-32 | Tensor size types?                                                            | `long` shape, strides, size and byte count; `int` `Ndim`; spans refuse more than `int.MaxValue` elements                                                                                                                                                  | Tensors over 2 GiB are legal data                                                                                                                 |
| SPC-34 | What does `foreach` over a Dict space yield?                                  | `KeyValuePair<string, Space>` (`IReadOnlyDictionary`); Python's `iter(d)` is `d.Keys`                                                                                                                                                                     | The BCL and LINQ contract                                                                                                                         |
| SPC-35 | Which types convert to a Box `Bound`?                                         | Exactly `bool`, signed integers (Int), unsigned integers (Uint), `float`/`double` (Float)                                                                                                                                                                 | Rust's `From` set; literals bind exactly                                                                                                          |
| ENV-39 | What is `EnvFactory.ServeAsync`?                                              | `rlmesh_ext_env_factory_start` (X-13) run synchronously on the caller's thread, then `WaitAsync`; the caller's `Cancel` watched by the server for the whole serve, the `CancellationToken` acting through `Shutdown()`; no signal handlers                | No async or engine serve path may install process-wide signal handlers; both serve twins fail startup before returning a task                     |
| MDL-23 | How is a policy authored?                                                     | `IPolicy` plus eight corner interfaces; both forms of one corner refused; `ILoad`, `IOnEpisodeEnd`, `IClose`, `ILoadedNativeChunk`; `ModelDeclarations` with `{ get; set; }` properties, copied when read; `Model.FromConfig(IPolicy, Metadata)`          | AOT-safe, `MonoBehaviour`-friendly, mirrors C's slots and static fields; one options convention (D-22)                                            |
| MDL-24 | Does `Model.ServeAsync` install signal handlers?                              | No: the sync model serve on a pool thread with `no_signal_handlers` set (X-15) and the twin's token in `cancel` (a set caller `Cancel` plus a cancellable token is refused until v2); `Ext.ModelServeOptions.NoSignalHandlers` for an engine's own thread | No async or engine serve path may install process-wide signal handlers; a field maps onto the facade's existing switch                            |
| MDL-25 | `PredictContext` lifetime and state storage?                                  | Lent for one corner call (later use is `ObjectDisposedException`); state values are `object?` behind a `GCHandle`                                                                                                                                         | C's lending rule; never reads a freed store                                                                                                       |
| RUN-31 | How do authored envs become targets (no conversions from interfaces, CS0552)? | `EnvTarget` converts from classes only; `EnvTarget.Local/Vector/Factory` for authored objects, plus an `IEnv` overload on every entry point, each through `Env.For` (ENV-37)                                                                              | Rust's one blanket conversion plus named kinds                                                                                                    |
| RUN-32 | Async twin of `new Session(...)`?                                             | `RLMeshApi.SessionAsync(...)`                                                                                                                                                                                                                             | A constructor cannot be async                                                                                                                     |
| RUN-33 | Properties that connect or fail on I/O?                                       | Methods with twins (`Session.Contract()`, `SelectedWorkflowEdition()`)                                                                                                                                                                                    | .NET properties should be cheap; a property has no twin                                                                                           |
| RUN-34 | Where do recorder records and `SCHEMA` go?                                    | Root namespace; `Recorder.Schema`                                                                                                                                                                                                                         | `RLMesh.Recorder` cannot be both a namespace and a type                                                                                           |
| RUN-35 | Are `StepEvent.read` / `FrameSources.render` delegates?                       | Sealed `ReadFn` / `RenderFn` classes with `Invoke` and reference equality                                                                                                                                                                                 | Delegate equality is not Python's identity and would leak into record equality                                                                    |
| RUN-36 | Hooks and `Recorder.Capture`'s return?                                        | `abstract class RunHooks : IRunHooks, IDisposable`; options take `IRunHooks`; `Capture` returns `RunHooks`                                                                                                                                                | `using var capture = rec.Capture(...)` with no cast                                                                                               |
| RMT-24 | Async construction of remote clients?                                         | Sync constructor connects; `static Task<RemoteEnv> ConnectAsync(options, ct)`                                                                                                                                                                             | Constructors cannot be async; matches Rust's `new_async`                                                                                          |
| RMT-25 | Client accessors?                                                             | Properties captured at construction over owned copies; only `SelectedWorkflowEdition` reads C each time                                                                                                                                                   | C's borrowed pointers die at `_free`                                                                                                              |
| RMT-26 | Re-implement the `RLMESH_API_KEY` lookup in C#?                               | No; C reads it; an in-process key goes in `ApiKey`                                                                                                                                                                                                        | CoreCLR on Unix does not propagate `Environment.SetEnvironmentVariable` to C; one resolution path                                                 |
| ADP-30 | Python's union-valued adapter fields?                                         | Small immutable types with Rust's names (`Normalize`, `AxisValue`, `StateEncoding`, …), factories and implicit conversions                                                                                                                                | Compile-time checks; object initializers read like Python's keywords                                                                              |
| ADP-31 | Keep colliding leaf names (`Action`, `Image`, `Text`)?                        | Keep; docs show `using Action = RLMesh.Adapters.Action;`                                                                                                                                                                                                  | R-4: one name per symbol in every language                                                                                                        |
| ADP-32 | Custom adapter shape?                                                         | `IAdapterBase` plus optional `IReset`/`IObserve`/`IExplain`; `abstract class AdapterBase`; explain text from `GetType().Name`                                                                                                                             | Python's explain text; metadata-only, AOT-safe                                                                                                    |

## C ABI additions for v0.2

C# needs no C ABI change for parity with Python: the spike bound ABI 5 from
C# on the tested path (spaces, env vtable, env server, Python driving a
C#-served env end to end), and every area maps its rows onto existing ABI 5
calls. Five additive changes serve v0.2's real consumers: C++ engine hosts
(an Unreal plugin's session, remote-client, factory-serve and model-serve
calls on its own threads), the C# foundation's cancellation plumbing and
tests, and factory serves that report their own runtime. ABI 5 is
unreleased, so each lands in the ABI 5 generation with no bump; struct
fields are appended under R-19, after the ABI-5 `sizeof`, as
`RLMeshRemoteEnvOptions.api_key` was. All five are **pending user
approval** ([table](#c-decisions-pending-user-review)); C++ gains each one
in the same release, and Rust gains what it lacks (X-16 only). A rejected
addition drops the feature that needs it from v0.2 (R-16), not its safety.

### X-12 Per-call cancel inputs

```c
/* Fenced (R-25). Each is its unfenced call plus `cancel`: NULL = exactly the
 * unfenced call; otherwise borrowed for the call and watched beside the
 * handle's own token, subscribed once at admission. */
RLMESH_API RLMeshStatus rlmesh_ext_session_reset(RLMeshSession* session, const int64_t* seed,
                                                 const int64_t* trial_index,
                                                 const RLMeshExtCancelToken* cancel,
                                                 RLMeshValue** out_observation,
                                                 RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_ext_session_predict(RLMeshSession* session,
                                                   const RLMeshValue* observation,
                                                   const RLMeshExtCancelToken* cancel,
                                                   RLMeshValue** out_action);
RLMESH_API RLMeshStatus rlmesh_ext_session_step(RLMeshSession* session, const RLMeshValue* action,
                                                const RLMeshExtCancelToken* cancel,
                                                RLMeshValue** out_observation, double* out_reward,
                                                bool* out_terminated, bool* out_truncated,
                                                RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_ext_remote_env_reset(RLMeshRemoteEnv* remote_env,
                                                    const int64_t* seed, const RLMeshValue* options,
                                                    const RLMeshExtCancelToken* cancel,
                                                    RLMeshValue** out_observation,
                                                    RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_ext_remote_env_step(RLMeshRemoteEnv* remote_env,
                                                   const RLMeshValue* action,
                                                   const RLMeshExtCancelToken* cancel,
                                                   RLMeshValue** out_observation, double* out_reward,
                                                   bool* out_terminated, bool* out_truncated,
                                                   RLMeshValue** out_info);
RLMESH_API RLMeshStatus rlmesh_ext_remote_vector_env_reset(
    RLMeshRemoteVectorEnv* remote_vector_env, const int64_t* seed, size_t seed_len,
    const RLMeshValue* options, const RLMeshExtCancelToken* cancel,
    RLMeshValue** out_observation, RLMeshValue** out_info);
/* A short capacity fails before I/O and leaves ALL outputs untouched. */
RLMESH_API RLMeshStatus rlmesh_ext_remote_vector_env_step(
    RLMeshRemoteVectorEnv* remote_vector_env, const RLMeshValue* actions,
    const RLMeshExtCancelToken* cancel, RLMeshValue** out_observation, double* out_rewards,
    bool* out_terminated, bool* out_truncated, size_t capacity, size_t* out_len,
    RLMeshValue** out_info);
```

- **Contract** (R-18 "Per-call cancel"). Each variant behaves as its
  unfenced call, which stays as it is. A non-NULL `cancel` is watched at
  exactly the points where the call watches its handle's `cancel_handle()`
  token; the call fails with R-21's cancellation class when either fires,
  with R-18's single-call state rule (the episode stays open, later calls
  are allowed). The token is borrowed: the caller may free it once the
  call returns.
- **Subscribed once, at admission.** The call subscribes its handle token
  and `cancel` together, once, after native admission: after the NULL,
  closed, busy and overlap checks pass (the capi handle's `try_lock`,
  `Session::call`'s `context.busy()` in `rs/run/session.rs`) and before any
  env, model or network work. A call refused at admission subscribes
  nothing, so a firing pending on its token stays pending for that token's
  next call (R-18's idle `cancel()`); subscribing earlier would lose it,
  because `CancelToken::subscribe` takes the pending flag (`rs/ext.rs`).
- **Carried explicitly.** The subscription is made on the calling thread and
  moved into the call's own watcher (`CancelWatch`, `rs/run/drive.rs`),
  which the call's tokio work owns until it returns. A firing between
  admission and the watcher's first poll is not lost: the subscription
  records the token's generation when it is taken. Nothing is read from or
  left in thread-local state, so a worker thread reused for another call
  carries nothing over, and a callback sees no caller token.
- **Released on return**, on success, failure or cancel. A firing after
  that reaches nothing, so a per-operation token has no completion race; a
  second call refused at admission never subscribed, so firing its token
  cannot reach the first call (no admission race). A C caller that passes
  one token to two concurrent calls cancels both, as R-18 says of `cancel`
  fields.
- **Why these seven, explicit.** Every other cancellable call already has
  an explicit input: `rlmesh_run`, `rlmesh_model_run`, the session
  constructors, `rlmesh_session_run`, `rlmesh_env_factory_serve`,
  X-13's start and `rlmesh_model_serve` carry a `cancel` options field. The
  single calls with fixed signatures are the gap, and v0.2's consumers need
  only these seven: an engine host stepping a session or a remote env from a
  worker and cancelling that one step, and the C# foundation's cancellation
  plumbing (R-18's C# bullet; its tests drive the remote variants against a
  C#-served env). Other handle calls (`contract`, the edition reads, remote
  `render`, `close`, `shutdown`; dials stay uncancellable, RMT-6) keep only
  their handle token. Round 1's thread-local scope is withdrawn: it had to
  capture the subscription at admission and carry it through every watcher
  anyway (C# review round 2), so it only hid an explicit argument, while
  adding ambient state that callbacks and reused workers had to save and
  clear (Lessons, "no ambient TLS").
- **Implementation note.** The capi passes `cancel` to crate-private
  `binding` entry points beside the public facade calls: `Session::call`
  already takes `cancel: Option<CancelToken>` and chains it after the
  session's token (`rs/run/session.rs`; `reset`, `predict` and `step` pass
  `None` today); the remote clients' `cancellable` builds its `CancelWatch`
  from the handle token alone (`rs/env/client.rs`) and gains the extra
  token. No public Rust API changes (Rust cancels by dropping futures and
  through `cancel` fields), and `CallbackFrame` needs no change, since no
  thread state exists.
- **Sizing (R-19), conventions (R-20).** No struct and no `user_data`;
  inputs precede `cancel`, outputs follow it.
- **Consumers.** C++ `rlmesh::ext` overloads (below); the C# foundation
  (internal); C. C# design-record twins (R-16) would use them.

  ```cpp
  namespace rlmesh::ext {
  ResetResult reset(Session&, const CancelToken&, std::optional<int64_t> seed = {},
                    std::optional<int64_t> trial_index = {});
  Value predict(Session&, const CancelToken&, const Value& observation);
  StepResult step(Session&, const CancelToken&, const Value& action);
  // RemoteEnv& and RemoteVectorEnv&: their reset / step parameters after the
  // token, returning ResetResult, StepResult or VectorStepResult as the members do.
  }
  ```

- **Tests** (E-3 Cancellation, in C, C++ and C#), each driven by a scripted
  env blocking on a barrier: a `cancel` fired mid-call fails
  `rlmesh_ext_session_step` and leaves the session usable; a `cancel` fired
  after the call returned leaves the next call unaffected (completion); a
  second thread's overlapping call is refused with the RuntimeError class,
  its token then fired, and the first call completes normally (admission);
  a token fired before an overlapping call that is then refused still fails
  that token's next admitted call at its start (no lost firing); a token
  fired between admission and the watcher's start, through a test hook,
  cancels the call; a call that fails with another error releases its
  subscription; one worker thread running a call with token A and then a
  call with none is unaffected by firing A; the handle token and `cancel`
  each cancel when fired alone; a NULL `cancel` behaves as the unfenced
  call; the remote variants cancel a blocked remote `step` with the
  connection still usable.

### X-13 Background factory serve

```c
/* Fenced (R-25; Python's EnvFactory has no start). Steps 0-5 of
 * EnvFactory.serve, then rlmesh_env_server_start. */
RLMESH_API RLMeshStatus rlmesh_ext_env_factory_start(
    RLMeshEnvFactory* factory, const char* address,
    const RLMeshEnvFactoryServeOptions* options /* NULL = INIT */, RLMeshEnvServer** out);
```

- **Contract.** Runs steps 0 to 5 of Env's `EnvFactory.serve` on the
  calling thread, under the factory's exclusive admission (the factory's
  callbacks run there, ENV-34): the pre-fired `cancel` check, the branch
  check, `prepare`, `num_envs` `make_env` calls, the edition, the server
  with `tags` per ENV-18, the describe envelope published on the handshake
  (ENV-13) with X-14's host runtime, and the `RLMesh serving env on …`
  line. It then starts the server as `rlmesh_env_server_start` does (every
  lane on library threads, no signal handlers, ENV-11) and returns it in
  `*out`, owned by the caller. `rlmesh_env_server_wait`, `_shutdown`,
  `_address`, `_env_contract`, `_free` and
  `rlmesh_ext_env_server_describe_json` apply as to any server.
- **Nothing of the factory survives the return.** The call makes its own
  gate collection for the envs it makes, instead of the factory's one
  `serve_gates` slot that the foreground serve and a session's factory
  borrow use (`capi/env_factory.rs` `ServeDrain`, whose drop expects to own
  that slot): the `make` callback routes made envs into the collection of
  the call that is running (its `Scope` frame), not into the factory's
  data. On success the collection moves into the returned
  `RLMeshEnvServer`, as `output()` moves lanes, the vector env and
  `_claims` (`capi/env_server.rs`), and the server stops and closes those
  envs once when it is shut down or freed. Before returning, the call
  releases the factory's exclusive admission and pops its thread-local
  factory scope (`CURRENT`), so the caller may make, serve again or free
  the factory at once (never closed, as Python's `serve` never closes it),
  and nothing the server runs later calls the factory. Starting while the
  factory is busy (a foreground serve, a session borrowing it, another
  start in progress) is refused at admission with `RLMESH_ERR_RUNTIME`
  (ENV-34).
- **Cancel, for the whole serve.** A token already fired fails the call
  before the branch check, consuming the firing. During steps 1-5 the
  token has no subscriber, so a firing there sets its pending flag; the
  call checks it once the server is built, before starting it, and then
  shuts the built server down, closes the envs made so far and fails with
  the cancellation class (today's foreground serve detects the same firing
  at its watcher's first poll, `rs/env/factory.rs` `CancelWatcher`). On
  success the returned server owns a watcher holding its own reference to
  the token (a clone, so the caller may free its reference) for as long as
  it serves: a later firing stops the server (drain, close the made envs),
  and `rlmesh_env_server_wait` then reports the cancellation class once, as
  the foreground serve fails with it (ENV-33). The watcher stops with the
  server.
- **Failure.** On any failure `*out` is NULL, every env made so far is
  closed and freed exactly once, the factory's admission and scope are
  released, and the factory is usable.
- **Name.** `rlmesh_ext_` because Python's `EnvFactory` has no start (R-25);
  the user's working name `rlmesh_env_factory_start` would put a
  language-only extra on the shared surface.
- **Implementation note.** The facade has the pieces: the private
  `prepare_serve` that `serve_host` runs before `EnvServer::serve`
  (`rs/env/factory.rs`) plus `EnvServer::start`; the capi reaches them
  through a `binding` entry point beside `env_factory_native_serve`
  (`rs/binding.rs`), which also carries the watcher.
- **Sizing (R-19).** Reuses `RLMeshEnvFactoryServeOptions` (with X-14).
- **Consumers.** C++ `rlmesh::ext::start` (below; an Unreal host that keeps
  the factory envelope, ENV-13); C. C#'s `Start` and `EnvFactory.ServeAsync`
  are design record (ENV-39). Rust has `EnvFactoryExt::serve_async` and
  gains nothing.

  ```cpp
  EnvServer rlmesh::ext::start(EnvFactoryBase& factory, std::string_view address,
                               const EnvFactoryServeOptions& options = {});
  ```

- **Tests** (E-3 Serving, C and C++): a Python client drives a started
  factory server and reads the describe envelope a foreground
  `rlmesh_env_factory_serve` publishes; SIGINT and SIGTERM dispositions are
  unchanged while it serves (`sigaction` before and after); a pre-fired
  `cancel` runs no `prepare`; a firing during `prepare` fails the call
  after building, with every made env closed once; failure on the Nth
  `make`, a stamping failure, a bind failure and a start failure each
  leave `*out` NULL and close exactly the envs made; after success the
  factory is made from again, served in the foreground and freed while the
  server serves, and the server calls none of its callbacks (the factory's
  `current()` scope would panic); a second start while one is in progress
  is refused; a `cancel` fired while serving stops the server and `wait`
  reports the cancellation class; `rlmesh_env_server_shutdown` and `_free`
  close every made env once.

### X-14 Factory serve host runtime

```c
typedef struct RLMeshExtEnvFactoryServeOptions { /* fenced (R-25) */
  size_t struct_size;
  const char* host_runtime_json; /* NULL = legacy runtime (language "c"), as today */
} RLMeshExtEnvFactoryServeOptions;
/* As RLMESH_EXT_ENV_SERVER_OPTIONS_INIT: supplies the caller's RLMESH_HOST_RUNTIME_JSON;
 * on MSVC valid only for automatic storage. */
#define RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_INIT \
  {sizeof(RLMeshExtEnvFactoryServeOptions), RLMESH_HOST_RUNTIME_JSON}
#define RLMESH_EXT_ENV_FACTORY_SERVE_OPTIONS_MIN_SIZE sizeof(RLMeshExtEnvFactoryServeOptions)

typedef struct RLMeshEnvFactoryServeOptions {
  size_t struct_size;
  int64_t num_envs;               /* 1; <= 1 serves one env */
  const char* workflow_edition;   /* NULL/""/blank = None */
  const RLMeshValue* make_kwargs; /* DICT; NULL = {} */
  RLMeshExtCancelToken* cancel;   /* NULL = none, borrowed for the call */
  /* Appended after ABI 5: _reserved (0) starts exactly at the ABI-5 sizeof. */
  uint64_t _reserved;
  const RLMeshExtEnvFactoryServeOptions* ext; /* NULL = none: legacy runtime */
} RLMeshEnvFactoryServeOptions;
#define RLMESH_ENV_FACTORY_SERVE_OPTIONS_MIN_SIZE offsetof(RLMeshEnvFactoryServeOptions, _reserved)
/* _INIT appends 0, NULL. */
```

- **Contract.** `rlmesh_env_factory_serve` and `rlmesh_ext_env_factory_start`
  read `ext->host_runtime_json` with the validator that
  `RLMeshExtEnvServerOptions.host_runtime_json` uses (`capi/describe.rs`
  `host_runtime`): a JSON object whose every field is optional,
  `language` and `language_version` non-empty text, `framework_versions` an
  object of non-empty names and versions, `toolchain` checked only when
  present (exactly `name` and `version`, `name` gcc, clang or msvc,
  `version` `major.minor`), and any other field refused. An invalid value
  fails with `RLMESH_ERR_INVALID_ARGUMENT` before the branch check. The
  value is the handshake's PeerInfo and the published envelope's
  `runtime` host fields (COR-25), where today a factory serve always
  reports C's (`rs/binding.rs` `env_factory_native_serve`, language `"c"`).
  NULL `ext` or NULL `host_runtime_json` keeps today's legacy runtime.
- **Sizing (R-19).** The ABI-5 `sizeof` (40 bytes on 64-bit targets) stays
  the minimum; `_reserved` starts exactly there on every target, so the
  appended pointer never overlaps an ABI-5 caller's tail padding, and an
  ABI-5 caller's struct reads as `ext` NULL. `_reserved` must be zero, as in
  the structs that already check theirs (`capi/describe_api.rs`).
  `RLMeshExtEnvFactoryServeOptions` is new, so its minimum is its `sizeof`.
- **Consumers.** C++: `EnvFactoryServeOptions` gains no member; the wrapper
  always passes an `ext` carrying `RLMESH_HOST_RUNTIME_JSON`, as for env
  servers (ENV-12). C. C#'s `Ext.EnvFactoryServeOptions.HostRuntimeJson` is
  design record (ENV-38). Rust passes its own runtime and has no field.
- **Tests** (E-3 Serving): a C factory serve with NULL `ext` reports
  `"c"`, a C++ one `"c++"`; an invalid JSON value and an unknown field are
  refused before `prepare`; a struct of `_MIN_SIZE` is accepted with the
  legacy runtime; a nonzero `_reserved` is refused.

### X-15 Model serve without signal handlers

```c
typedef struct RLMeshExtModelServeOptions {
  size_t struct_size;
  const char* token;
  size_t predict_concurrency;
  bool has_predict_concurrency;
  uint8_t _reserved[sizeof(size_t) - 1];
  /* Appended after ABI 5; starts exactly at the ABI-5 sizeof. */
  bool no_signal_handlers;                /* false: install ENV-11's handlers, as today */
  uint8_t _reserved1[sizeof(size_t) - 1];
} RLMeshExtModelServeOptions;
#define RLMESH_EXT_MODEL_SERVE_OPTIONS_INIT \
  {sizeof(RLMeshExtModelServeOptions), NULL, 0, false, {0}, false, {0}}
#define RLMESH_EXT_MODEL_SERVE_OPTIONS_MIN_SIZE \
  offsetof(RLMeshExtModelServeOptions, no_signal_handlers)
```

- **Contract.** With `no_signal_handlers` true, `rlmesh_model_serve`
  installs no SIGINT / SIGTERM handler (ENV-11, MDL-20); it still blocks
  the calling thread and stops on its `cancel` token, the idle timeout, a
  remote shutdown or `rlmesh_model_close`. The facade already has this
  switch: `serve_blocking`'s `foreground` flag, false for
  `Model::serve_async` (`rs/model/facade.rs`). False, the zero value, keeps
  today's behaviour (R-19: zero is the default).
- **Why a field, not `rlmesh_model_start`.** No language has a model server
  handle: Python's `Model.serve` is foreground only and Rust's twin is a
  future. A start call would invent a handle type (address, wait,
  shutdown, free) with no Python counterpart for R-1 to map, while the
  field maps onto an existing facade switch, and a blocking serve on a
  worker thread with a `cancel` token already gives Rust's `serve_async`
  semantics (no handlers, stops on cancel).
- **Sizing (R-19).** The struct's `sizeof` is a multiple of `sizeof(size_t)`
  with no tail padding, so the field starts at the ABI-5 `sizeof`, which
  stays the minimum; an ABI-5 caller reads as false.
- **Consumers.** C++ `ext::ModelServeOptions::no_signal_handlers = false`
  (an engine host serving a model from its own thread); C. C#'s
  `Model.ServeAsync` is design record (MDL-24). Rust has `serve_async` and
  gains nothing.
- **Tests** (E-3 Serving, C and C++): with the field set, SIGINT / SIGTERM
  dispositions are unchanged while serving and a `cancel` stops the serve
  with the cancellation class; unset, the existing signal tests still pass;
  an ABI-5-sized struct reads as unset.

### X-16 Entrypoint unregistration

```c
/* Fenced (R-25). Removes entrypoint's registration only if it is expected's
 * (NULL = whatever is registered); routes already resolved keep theirs. */
RLMESH_API RLMeshStatus rlmesh_ext_unregister_entrypoint(const char* entrypoint,
                                                         const RLMeshObsTransform* expected,
                                                         bool* out_removed);
```

- **Contract.** Removes `entrypoint` from the process-wide registry that
  `rlmesh_ext_register_entrypoint` fills (MDL-11) when the registered
  transform is `expected`'s, by construction identity (the facade's
  `ObsTransform` compares by `Arc::ptr_eq`, `rs/adapters/callable.rs`), and
  releases the registry's reference; `expected` NULL removes whatever is
  registered. So a C++ plugin or a C# domain that unloads removes only what
  it registered, never a newer registration another owner made under the
  same name (the registry is one process-wide map, `ENTRYPOINTS` in
  `rs/adapters/execute.rs`). The check and the removal happen under one
  write lock; the removed value is dropped after the lock is released, so a
  transform's `free_user_data` that resolves or registers again cannot
  deadlock on it. `rlmesh_ext_register_entrypoint`'s replace path changes
  the same way (today the replaced value is dropped while the write guard
  is still held). The transform's `free_user_data` runs once its last
  reference (a resolved route of a live model or adapter, which holds its
  own clone) is released. `*out_removed` is false when the name was not
  registered or was registered by another transform; NULL `entrypoint` or
  `out_removed` is `RLMESH_ERR_INVALID_ARGUMENT`. `expected` is only
  compared; the caller keeps owning its handle.
- **Sizing (R-19), conventions (R-20).** No struct; no new `user_data`.
- **Consumers.** C++ (a plugin that unloads, such as an Unreal module) and
  Rust (the facade function the export wraps), below; C. The C# barrier
  unregisters every name it registered, by identity (D-43); the v0.2
  surface registers none.

  ```cpp
  bool rlmesh::ext::unregister_entrypoint(std::string_view name,
                                          const ObsTransform* expected = nullptr);
  ```

  ```rust
  // rlmesh::ext
  pub fn unregister_entrypoint(name: &str, expected: Option<&ObsTransform>) -> Result<bool>;
  ```

- **Tests:** register, resolve with `trust_entrypoints`, unregister: a new
  resolve fails with `AdapterResolutionError` as for an unknown name
  (ADP-11), the existing route keeps working, and `free_user_data` runs
  exactly once, after the route's model is freed; unregistering an unknown
  name returns false; owner A registers, owner B replaces, A's
  unregister by identity returns false and B's registration stays;
  resolution racing unregistration on another thread sees either the old
  transform or an unknown name, never a crash; a `free_user_data` that
  registers a name during the drop completes (no deadlock), for
  unregistration and for replacement alike.

The generator does not translate `_INIT` macros, `RLMESH_RANDOM_SAMPLE`'s
compound literal or `static inline` helpers (`rlmesh_abi_check` among them);
the hand layer supplies managed equivalents, and E-3 checks them field by
field against C (R-19, R-28). That needs no ABI change.

## Lessons for the v2 C ABI

Requirements the v2 core's C ABI must meet from its first draft, learned
from the C# design (R-28, X-11, D-43) and its two review rounds. They are
binding-agnostic: C#, C++ (Unreal), Python and later bindings all need them.
Concepts are the v2 draft's (`session-core-v0`): sessions, participants,
channels, samples, action chunks, leases, epochs and the edge runtime.

1. **Native-thread callbacks into managed runtimes.**
   - Every callback is a named C function-pointer typedef taking its
     `user_data` first, with a `free_user_data` beside it; no closures, no
     varargs, no by-value unions, no callback that must unwind. A
     generator then produces one delegate type per typedef, and static
     trampolines (`[MonoPInvokeCallback]`, IL2CPP, NativeAOT) suffice.
   - The ABI documents, per callback kind, the thread it runs on (the
     calling thread, one owned core thread, or any core thread, several at
     once). The core reuses a bounded set of threads: attaching a thread to
     Mono or IL2CPP has a cost, and a thread per call (per sample, per
     chunk) is not acceptable.
   - High-rate data never needs a reverse call per item: channel samples are
     also readable by polling a channel's latest value or a history window,
     and sample callbacks may deliver batches, so a 1 kHz channel does not
     mean 1,000 managed transitions a second.
2. **Game and main-thread dispatch** (Unity, Unreal, Godot).
   - Everything an engine must do on its main thread can be driven from
     that thread by a non-blocking call: an engine embodiment publishes
     samples and drains its applied-action queue from its tick, and the
     Gym view's lockstep or virtual tick is a call the engine makes, not a
     callback it must answer.
   - A callback that needs the main thread may complete later: it returns
     "pending" and the binding completes it through a completion call, so a
     core thread never blocks on the main thread.
   - When a call already runs on the thread a callback needs, the core
     calls it inline. No core call waits for a callback that can only run on
     the calling thread (the deadlock R-28 Threads describes).
   - The edge runtime's watchdog and fallback never wait on a binding: an
     Editor pause, a GC or a reload stalls the embodiment's managed side,
     and the edge holds and falls back on its own (lease lapse → hold → SAFE).
3. **Explicit per-operation cancellation.**
   - Every blocking or long-running call takes an operation context: a
     sized struct with any number of cancel sources and a deadline. No
     ambient thread-local cancel state (X-12's lesson).
   - The core subscribes the sources once, at admission, after its busy and
     overlap checks; a call refused at admission consumes no firing; the
     subscription is carried explicitly through every task and released on
     return. Several sources compose natively, so a binding never has to
     choose between the caller's token and its own (R-18's C# bullet).
   - Tokens fire from any thread without blocking. Session verbs (attach,
     lease request, reset, swap) each cancel on their own context; e-stop
     is never a cancellation and never queues behind one.
4. **Exception containment and error text.**
   - Callbacks return a status and report text, class and an opaque payload
     through an error object passed to the callback, not a thread-local
     slot: managed continuations change threads, and TLS errors forced
     R-28's "read on the same thread before any other call".
   - The core accepts any UTF-8 error text and repairs it (U+0000 and
     invalid sequences replaced) rather than refusing it, keeps the payload
     channel (in-process rethrow of the author's object), and has a
     reporting path that cannot fail (X-11's static fallback).
5. **Ownership of strings, tensors and samples.**
   - Every output is owned or borrowed by type, never by convention: owned
     text is one bytes type with one free; borrowed text states the call
     that invalidates it (R-5's split, which round 1 found missing).
   - Sample and tensor buffers are leases: an explicit acquire and release
     with a reference the binding owns, so no API returns memory whose
     lifetime is a wrapper's reachability (D-31). Zero-copy channel samples
     (shared memory) use the same lease.
   - Foreign-buffer import (DLPack) has a native state machine: the import
     either takes the descriptor or leaves it with the caller, and the core
     never runs the producer's deleter re-entrantly inside the import call
     (SPC-31's `Importing` state exists only because ABI 5 does).
6. **Consuming transfers.** A call that takes ownership takes it exactly on
   success and leaves it with the caller on failure, the same rule for
   callback outputs as for arguments, marked in the signature (ABI 5's
   factory `make` takes its outputs even on failure, `h:2758-2761`, which
   forced ENV-37's transfer state). Chunk and sample publication follow it.
7. **Editor reload and shutdown.**
   - A binding registers a host context (one per domain or load context),
     and every `user_data`, registry entry and subscription it creates
     belongs to it. Process-wide registries are keyed by owner and removed
     by identity (X-16).
   - `shutdown(host, deadline)`: closes admission for that host, cancels
     its operations, releases its registrations, and returns within the
     deadline with a report; it never blocks past it. Blocking teardown runs
     on core threads.
   - The core does the completion accounting: it counts reverse calls into
     the host in flight, decremented after each callback returns to native,
     and reports quiescence only when that count is zero and no native
     object holds the host's pointers (D-43 has to infer this by joining
     frees, step 6).
   - A preflight query (`quiescent(host)`) lets the host refuse or defer a
     reload; unload notifications are only a fallback. A host shutdown is a
     participant detach: the session host sees its liveliness end, its
     leases lapse and the edge falls back, so an Editor reload is a clean
     detach, not a fault.
8. **Signal handlers.** The core installs no process-wide signal handler
   unless a console host asks for it with an explicit call; library mode,
   the default, installs none (ABI 5's foreground default needed X-13 and
   X-15 to opt out).
9. **Describe and host runtime.** The binding reports its runtime once, at
   host registration: language, runtime (CoreCLR, Mono, IL2CPP), versions
   and scripting backend as free-form text, with no language-specific
   validation such as ABI 5's `toolchain` check (ENV-38). It travels in
   the participant's identity and the session's evidence log.
