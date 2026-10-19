# Language parity

Status: Reviewed (Opus rounds 2–6); user decisions settled 2026-10-15; implemented in Python, Rust, C (ABI 5) and C++17 (tested also at C++20; D-5 amended 2026-10-16). Draft: C# column, C# review round 1 applied 2026-10-20, pending user review (D-17; not implemented, G-29), with five C ABI additions pending approval (X-12 to X-16).

RLMesh is one library in five languages: Python (the reference), Rust (the
`rlmesh` facade crate), C (`crates/rlmesh-capi/include/rlmesh.h`), C++
(`rlmesh.hpp`, wrapping C one to one) and C# (the `RLMesh` package in
`csharp/`, also wrapping C one to one; D-17). C# targets game engines and
simulators; engine integrations such as `integrations/unity/` sit on top of
it and are outside this spec, though their constraints shape it (R-28). If you know the Python API you know
what to expect in every other language, adjusted only by the fixed
translation rules. A difference that no rule explains is a bug.

Python's Stable surface
(`python/rlmesh/tests/api_surface/snapshots/api_surface.json`, labels in
[compatibility](../compatibility.md)) defines the nouns, verbs, defaults and
field names. The wire contract (`crates/rlmesh-proto/proto/`) is frozen and
this spec does not change it.

## How to read

1. Start with [rules](language-parity/rules.md): the translation table, the
   options-struct and async rules, C struct sizing, the one error matrix,
   warnings, fencing, decisions, Python changes, shapes, known gaps and the
   enforcement. C# rules are clauses inside each rule, plus R-28 (C# runtime
   constraints), X-11 (C# trampolines) and X-12 to X-16 (the C ABI
   additions C# needs, detailed [below](#c-abi-additions-for-v02)).
2. Then read the area you work on. Each area maps every Stable Python symbol
   it owns to Rust, C, C++ and C#, and lists its own Python changes and current
   differences. A symbol shared by several areas has one owner (rules R-27);
   the others link to it.
3. IDs are stable: `R-n` rules, `D-n` decisions, `P-n` Python changes, `X-n`
   cross-cutting changes, `G-n` gaps, `E-n` enforcement. Area files use
   `<AREA>-n` for decisions and `<AREA>-P<n>` for Python changes (`COR`, `SPC`,
   `ENV`, `MDL`, `RUN`, `RMT`, `ADP`). Cite them in code review and commit
   messages.
4. "User (accepted default)" marks a default the user accepted on
   2026-10-15. Every one is listed under
   [Decisions accepted by the user](#decisions-accepted-by-the-user).
   "Pending (C#)" marks a C# default awaiting review; every one, and each
   pending C ABI addition, is listed under
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

D-17 (C# at full parity, wrapping C) is the user's. Every other C# choice
below is a recommended default, written into the rules and area tables as
"Pending (C#)", revised after C# review round 1 (rules' Review notes). The
X-12 to X-16 rows are additive C ABI changes, detailed in
[C ABI additions for v0.2](#c-abi-additions-for-v02), each pending user
approval; D-44's test mechanism is a pending user decision, not a
recommendation. Accepting one makes it settled; rejecting one is a spec
change in the cited rule or area.

| ID     | Question                                                                      | Recommended default                                                                                                                                                                                                                                                                    | Rationale                                                                                                                                                                                                   |
| ------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-18   | How is the C# library laid out and generated?                                 | `csharp/` at the root, package and namespace `RLMesh`; raw layer generated by ClangSharpPInvokeGenerator 21.1.8.4 (the spike's flags, `internal`), with a `--check` staleness gate in `mise run check`; hand-written public layer                                                      | The spike generated 4.5k lines with no `delegate*` and bound ABI 5 with no changes; a generator keeps the 510 imports in step with `rlmesh.h`                                                               |
| D-19   | Which targets and language level?                                             | `netstandard2.1` + `net10.0`; public surface consumable from C# 9; 64-bit only; tests on the .NET 10 runtime; E-2 diffs both assemblies against an allowlist (`System.Half` only); SDK pinned by `global.json`, installed by mise                                                      | .NET 8 support ends 2026-11-10; Unity, Mono and Godot need netstandard2.1 and C# 9; the generated layout is per-architecture                                                                                |
| D-20   | How are names cased?                                                          | One transform: `PascalCase` members and constants, `camelCase` parameters, `@` for keyword parameters, no trailing `_`                                                                                                                                                                 | .NET conventions while E-2 stays one mapping per name; `@params` keeps the metadata name equal to Python's                                                                                                  |
| D-21   | Where do free functions live?                                                 | `RLMesh.RLMeshApi`, `RLMesh.Spaces.SpacesApi`, `RLMesh.Adapters.AdaptersApi` (`<Module>Api`); type-named builders become constructors; classmethods stay static on their type                                                                                                          | A class cannot share its namespace's name; a bare `Api` collides with user types; `using static RLMesh.RLMeshApi;` reads like Python; mirrors D-6                                                           |
| D-22   | Options classes or C# optional parameters for keywords?                       | Sealed classes with `{ get; set; }` properties initialized to Python's defaults, `options = null` last; not records, not `init`-only                                                                                                                                                   | Optional parameters bake constant defaults into callers; records add unreviewed equality; `init` blocks conditional assembly and C# 9 has no `required`. Challenges the `init` starting point               |
| D-23   | How do R-7's positional calls look?                                           | Optional parameters defaulting to `null` (= Python's default) or Python's constant for a non-optional scalar                                                                                                                                                                           | `null` bakes in nothing that can change; named arguments read like Python                                                                                                                                   |
| D-24   | Async twins?                                                                  | `<Name>Async` for exactly Rust's twins, `async` methods over the sync C call on a library worker (one per exclusive handle, a growing pool otherwise); per-operation cancel scope (X-12); sync calls never refused in async code                                                       | Engines must not block a frame; no thread per step; no completion or admission race; `await` rethrows the same `CancelledError`                                                                             |
| D-25   | Which .NET bases do the exceptions use?                                       | `ValueError : ArgumentException`, `TypeError : InvalidCastException`, `RuntimeError : InvalidOperationException`, `OSError : IOException`, `Ext.CancelledError : OperationCanceledException`; documented: every rlmesh failure class is an `InvalidOperationException`                 | Keeps .NET idioms while copying Python's tree; `catch (InvalidOperationException)` reads as `except RuntimeError`                                                                                           |
| D-26   | Equality and hashing?                                                         | `IEquatable<T>`/`Equals`/`==` where Python has `__eq__`; `GetHashCode` always agrees, content-based where Python hashes (floats included), coarse where it does not                                                                                                                    | .NET requires hash to agree with equality; .NET doubles are reflexive, so Rust's float restriction does not apply                                                                                           |
| D-27   | Collection types?                                                             | Inputs `IReadOnlyList<T>` / spans, copied; `params T[]` for `*args`; outputs `IReadOnlyList<T>`; record list fields `EquatableList<T>`                                                                                                                                                 | Record equality on a plain list is reference equality, which is not Python's `==`                                                                                                                           |
| D-28   | Frozen dataclasses?                                                           | `sealed record` classes with an explicit `camelCase` constructor for required fields, `init` properties for defaulted ones, `ToString()` = Python's `str`, `With<Name>` instance methods                                                                                               | `with` and value equality for free; positional-record parameters would be `PascalCase` and fail E-2                                                                                                         |
| D-29   | Tuples and Gymnasium results?                                                 | Hand-written `readonly struct ResetResult` / `StepResult` (and vector forms) with named properties and `Deconstruct`; other tuples named `ValueTuple`s                                                                                                                                 | `var (obs, reward, …) = s.Step(a)` reads like Gymnasium and `result.Reward` stays available                                                                                                                 |
| D-30   | Python aliases (`InfoDict`, `Kwargs`, `StackSpec`)?                           | Spelled as their target (`Metadata`, `IReadOnlyList<int>`)                                                                                                                                                                                                                             | .NET has no exported alias; a subclass breaks argument binding                                                                                                                                              |
| D-31   | Tensor memory?                                                                | No span over native memory on the shared surface: `Buffer()` returns an owning `TensorLease<byte>` (own native reference, no finalizer); typed leases and fenced `DangerousAsSpan<T>` in `RLMesh.Ext`; inputs copied                                                                   | A span does not keep its tensor alive: no use-after-free on the safe path, a forgotten lease leaks instead                                                                                                  |
| D-32   | Authoring shape?                                                              | Required interface per protocol, one-member optional interfaces detected with `is`, declarations objects (MDL-23, ENV-36), no base class required                                                                                                                                      | A `MonoBehaviour` or Godot `Node` can implement it; a default interface method cannot express C's "not defined" and IL2CPP support for it is unestablished; no static abstracts in netstandard2.1; AOT-safe |
| D-33   | Fence?                                                                        | `RLMesh.Ext` namespace; raw layer `internal`, no public binding module, no `InternalsVisibleTo` for integrations                                                                                                                                                                       | Mirrors C++ `rlmesh::ext`; the Unity layer proves the public surface is enough                                                                                                                              |
| D-34   | Integer types?                                                                | C's widths (`long`, `ulong`, `uint`, `ushort`); `int` for in-memory counts and indices; `[CLSCompliant(false)]`                                                                                                                                                                        | One type per E-2 comparison; BCL collections count in `int`                                                                                                                                                 |
| D-35   | Durations?                                                                    | `TimeSpan`, `_seconds` dropped, 100 ns precision; `Timeout.InfiniteTimeSpan` is negative, not infinite                                                                                                                                                                                 | Idiomatic; R-14's checks stay in C                                                                                                                                                                          |
| D-36   | A string with U+0000 bound for a NUL-terminated C input?                      | Refused with the ValueError class before the call (unpaired surrogates too); C++ makes the same refusal in the same release; exception text reported to C is sanitized, not refused (X-11)                                                                                             | Silent truncation (C++ today) hides bugs; the two wrappers over C agree                                                                                                                                     |
| D-37   | Warning handler shape?                                                        | One `Action<Warning>?` via `RLMeshApi.SetWarningHandler`, not an `event`                                                                                                                                                                                                               | R-23 is one atomic handler; several subscribers make "the handler failed" ambiguous                                                                                                                         |
| D-38   | Callback threading?                                                           | Where C runs them: the calling thread for local sync runs and sessions and for factory callbacks, library threads (several at once) for served envs and models; never marshalled                                                                                                       | An integration must know when to call inline; queuing an inline call to the main thread deadlocks                                                                                                           |
| D-39   | Finalizers?                                                                   | `SafeHandle` safety net; `Dispose()` frees inline and returns after the native free; a finalizer-reached callback-bearing or blocking handle is freed on one cleanup thread by a record that owns its dependency references                                                            | The finalizer thread must not block or reenter authored code; dependencies outlive the native free                                                                                                          |
| D-40   | Distribution?                                                                 | NuGet with `runtimes/<rid>/native` (64-bit; no WebGL); Unity consumes the prebuilt netstandard2.1 DLL (plus an `__Internal` build for iOS); host runtime `{"language":"csharp","framework_versions":{…}}` through `HostRuntimeJson`, set by the Unity integration                      | Frees the implementation from Unity's compiler version; C refuses a `toolchain` other than gcc, clang or msvc; detection cannot tell IL2CPP from Mono                                                       |
| D-41   | A call on a disposed wrapper?                                                 | `ObjectDisposedException`                                                                                                                                                                                                                                                              | A C#-only state, idiomatic, and an `InvalidOperationException` like `RuntimeError`                                                                                                                          |
| D-42   | Enforcement?                                                                  | `csharp` manifest column with overloads by documentation ID; System.Reflection.Metadata tool over both assemblies with IL checks; fixtures on the .NET 10 JIT against each assembly, NativeAOT as a supplement; executed generated defaults test                                       | Reads the shipped assemblies without loading them; NativeAOT is not evidence for Unity (D-44)                                                                                                               |
| D-43   | How does C# survive a Unity domain reload?                                    | `Ext.Lifetime.Shutdown(timeout)`: refuse work, fail pending dispatch, cancel and join operations, release every handle holding a managed pointer (X-16 for entrypoints), wait for every native free callback and the cleanup queue; `false` on timeout                                 | Covers every holder of a managed pointer and proves the drain through `free_user_data`; no native quiesce unless D-44 shows a hang                                                                          |
| D-44   | When may v0.2 claim Unity support, and how are the Unity tests run?           | Gate: an IL2CPP player smoke test and a Mono Editor domain-reload test pass before v0.2 claims Unity support. Mechanism (Unity licence, manual run or dispatch-only CI): **pending user decision**                                                                                     | NativeAOT and Mono-on-.NET runs miss IL2CPP codegen, UnityLinker stripping and the Editor's reload                                                                                                          |
| COR-28 | How does C# describe a type?                                                  | `RLMeshApi.Describe<TPolicy>(ModelDeclarations? declarations = null, …) where TPolicy : IPolicy` (corners from implemented interfaces); a `TPolicy` implementing `IModelDeclarations` with `null` declarations is TypeError; no env-factory type form                                  | No static interface members or reflection; never silently drops declarations; C also describes a factory only as a handle                                                                                   |
| COR-29 | How is a C# `Value` built?                                                    | An exact implicit conversion per type: `bool`, `sbyte`, `byte`, `short`, `ushort`, `int`, `uint`, `long`, `float`, `double`, `string`, `Tensor`, `Metadata`; `char` and `ulong` conversions are `[Obsolete(error: true)]`; `Count`/indexer/`TryGet` without a collection interface     | Exact operators make every literal and integer type bind one operator (no CS0457); counterpart of C++'s deleted overloads                                                                                   |
| COR-30 | What is C# `Metadata`?                                                        | Managed, mutable, insertion-ordered `IDictionary<string, Value>`; order-insensitive `Equals`/`==`; hash on `Count`                                                                                                                                                                     | Python `dict` equality, knowingly against .NET's advice for mutable types                                                                                                                                   |
| SPC-31 | How is tensor memory read and written?                                        | Constructors copy; `Buffer()` is an owning `TensorLease<byte>`; typed `Lease<T>`/`ToArray<T>`, raw `DangerousAsSpan<T>` and `Ext.DlpackTensor` in `RLMesh.Ext`; a foreign DLPack deleter never runs on the finalizer thread                                                            | R-28 lifetime safety; matches Rust's `TensorExt` and C++'s `ext`                                                                                                                                            |
| SPC-32 | Tensor size types?                                                            | `long` shape, strides, size and byte count; `int` `Ndim`; spans refuse more than `int.MaxValue` elements                                                                                                                                                                               | Tensors over 2 GiB are legal data                                                                                                                                                                           |
| SPC-33 | Space class hierarchy?                                                        | Unsealed `Space` with sealed `Box`/`Discrete`/`Dict`/Experimental kinds; `AsBox()`-style downcasts; cached child wrappers; `Spaces.Tuple` keeps its name                                                                                                                               | `is Box` works as Python's `isinstance`                                                                                                                                                                     |
| SPC-34 | What does `foreach` over a Dict space yield?                                  | `KeyValuePair<string, Space>` (`IReadOnlyDictionary`); Python's `iter(d)` is `d.Keys`                                                                                                                                                                                                  | The BCL and LINQ contract                                                                                                                                                                                   |
| SPC-35 | Which types convert to a Box `Bound`?                                         | Exactly `bool`, signed integers (Int), unsigned integers (Uint), `float`/`double` (Float)                                                                                                                                                                                              | Rust's `From` set; literals bind exactly                                                                                                                                                                    |
| ENV-36 | How are envs and factories authored?                                          | `IEnv`/`IVectorEnv` plus `IRender`, `IClose`, `IRenderMode`, `IMetadata`, `IId`; `IEnvFactory` plus `IPrepare`, `ITagsFor`, `IEnumerateVariants` and `EnvFactoryDeclarations`; public `Env`/`VectorEnv`/`EnvFactory` handles                                                           | R-12's pattern; no base class; describe by handle as C does                                                                                                                                                 |
| ENV-37 | One native handle per authored env?                                           | `Env.For` / `VectorEnv.For` only (no public constructor); a lock-protected identity registry with a weak wrapper reference, retired only when the native handle's `free_user_data` runs; API-created handles disposed by their API; a factory `Make` result is transferred, not shared | No double native handle under races, no root chain keeping an undisposed `Env` alive, no second handle while a server still owns the first                                                                  |
| ENV-38 | How does a served C# env report its runtime?                                  | `HostRuntimeJson` on `Ext.EnvServerOptions` and (X-14) `Ext.EnvFactoryServeOptions`; `null` = `{"language":"csharp","framework_versions":{"dotnet" or "mono": version}}`                                                                                                               | C accepts no `toolchain` for C#; a prebuilt assembly cannot tell IL2CPP from Mono, so the Unity integration sets it                                                                                         |
| ENV-39 | What is `EnvFactory.ServeAsync`?                                              | `Ext.EnvFactoryExtensions.Start` (X-13: pre-serve steps, envelope published, started `EnvServer`) plus `WaitAsync`; no signal handlers; dropped from v0.2 if X-13 is rejected                                                                                                          | No async or engine serve path may install process-wide signal handlers                                                                                                                                      |
| ENV-40 | How does a `CancellationToken` act on server twins?                           | `ServeAsync` checks the token before `Start()` (a bind failure throws before the task is returned); cancel triggers `Shutdown()` on the twin's worker, not in the token callback, and ends `Canceled`; `WaitAsync` waits in 100 ms slices; `ShutdownAsync` cancel stops waiting only   | C# form of ENV-32's dropped futures; never starts a cancelled server and never blocks inside `Cancel()`                                                                                                     |
| MDL-23 | How is a policy authored?                                                     | `IPolicy` plus eight corner interfaces; both forms of one corner refused; `ILoad`, `IOnEpisodeEnd`, `IClose`, `ILoadedNativeChunk`; `ModelDeclarations` with `{ get; set; }` properties, copied when read; `Model.FromConfig(IPolicy, Metadata)`                                       | AOT-safe, `MonoBehaviour`-friendly, mirrors C's slots and static fields; one options convention (D-22)                                                                                                      |
| MDL-24 | Does `Model.ServeAsync` install signal handlers?                              | No: the sync model serve on a pool thread with `no_signal_handlers` set (X-15); `Ext.ModelServeOptions.NoSignalHandlers` for an engine's own thread; dropped from v0.2 if X-15 is rejected                                                                                             | No async or engine serve path may install process-wide signal handlers; a field maps onto the facade's existing switch                                                                                      |
| MDL-25 | `PredictContext` lifetime and state storage?                                  | Lent for one corner call (later use is `ObjectDisposedException`); state values are `object?` behind a `GCHandle`                                                                                                                                                                      | C's lending rule; never reads a freed store                                                                                                                                                                 |
| RUN-31 | How do authored envs become targets (no conversions from interfaces, CS0552)? | `EnvTarget` converts from classes only; `EnvTarget.Local/Vector/Factory` for authored objects, plus an `IEnv` overload on every entry point, each through `Env.For` (ENV-37)                                                                                                           | Rust's one blanket conversion plus named kinds                                                                                                                                                              |
| RUN-32 | Async twin of `new Session(...)`?                                             | `RLMeshApi.SessionAsync(...)`                                                                                                                                                                                                                                                          | A constructor cannot be async                                                                                                                                                                               |
| RUN-33 | Properties that connect or fail on I/O?                                       | Methods with twins (`Session.Contract()`, `SelectedWorkflowEdition()`)                                                                                                                                                                                                                 | .NET properties should be cheap; a property has no twin                                                                                                                                                     |
| RUN-34 | Where do recorder records and `SCHEMA` go?                                    | Root namespace; `Recorder.Schema`                                                                                                                                                                                                                                                      | `RLMesh.Recorder` cannot be both a namespace and a type                                                                                                                                                     |
| RUN-35 | Are `StepEvent.read` / `FrameSources.render` delegates?                       | Sealed `ReadFn` / `RenderFn` classes with `Invoke` and reference equality                                                                                                                                                                                                              | Delegate equality is not Python's identity and would leak into record equality                                                                                                                              |
| RUN-36 | Hooks and `Recorder.Capture`'s return?                                        | `abstract class RunHooks : IRunHooks, IDisposable`; options take `IRunHooks`; `Capture` returns `RunHooks`                                                                                                                                                                             | `using var capture = rec.Capture(...)` with no cast                                                                                                                                                         |
| RMT-24 | Async construction of remote clients?                                         | Sync constructor connects; `static Task<RemoteEnv> ConnectAsync(options, ct)`                                                                                                                                                                                                          | Constructors cannot be async; matches Rust's `new_async`                                                                                                                                                    |
| RMT-25 | Client accessors?                                                             | Properties captured at construction over owned copies; only `SelectedWorkflowEdition` reads C each time                                                                                                                                                                                | C's borrowed pointers die at `_free`                                                                                                                                                                        |
| RMT-26 | Re-implement the `RLMESH_API_KEY` lookup in C#?                               | No; C reads it; an in-process key goes in `ApiKey`                                                                                                                                                                                                                                     | CoreCLR on Unix does not propagate `Environment.SetEnvironmentVariable` to C; one resolution path                                                                                                           |
| ADP-30 | Python's union-valued adapter fields?                                         | Small immutable types with Rust's names (`Normalize`, `AxisValue`, `StateEncoding`, …), factories and implicit conversions                                                                                                                                                             | Compile-time checks; object initializers read like Python's keywords                                                                                                                                        |
| ADP-31 | Keep colliding leaf names (`Action`, `Image`, `Text`)?                        | Keep; docs show `using Action = RLMesh.Adapters.Action;`                                                                                                                                                                                                                               | R-4: one name per symbol in every language                                                                                                                                                                  |
| ADP-32 | Custom adapter shape?                                                         | `IAdapterBase` plus optional `IReset`/`IObserve`/`IExplain`; `abstract class AdapterBase`; explain text from `GetType().Name`                                                                                                                                                          | Python's explain text; metadata-only, AOT-safe                                                                                                                                                              |
| X-12   | C ABI: an operation-scoped cancel input?                                      | `rlmesh_ext_cancel_scope_enter` / `_exit`; C++ `ext::CancelScope`. **Pending user approval**                                                                                                                                                                                           | Removes both races for every single-call twin with two exports instead of ~25 variants                                                                                                                      |
| X-13   | C ABI: a background factory serve?                                            | `rlmesh_ext_env_factory_start(factory, address, options, &server)` publishing the envelope; C++ `ext::start`. **Pending user approval**                                                                                                                                                | Engine hosts keep the factory envelope (ENV-13) without signal handlers                                                                                                                                     |
| X-14   | C ABI: a host runtime for factory serves?                                     | `RLMeshEnvFactoryServeOptions.ext` → `RLMeshExtEnvFactoryServeOptions.host_runtime_json`. **Pending user approval**                                                                                                                                                                    | A C++ or C# factory serve reports language `"c"` today                                                                                                                                                      |
| X-15   | C ABI: model serve without signal handlers?                                   | `RLMeshExtModelServeOptions.no_signal_handlers` (zero = install), not `rlmesh_model_start`. **Pending user approval**                                                                                                                                                                  | No language has a model server handle; the facade already has the switch                                                                                                                                    |
| X-16   | C ABI: unregister a trusted entrypoint?                                       | `rlmesh_ext_unregister_entrypoint(name, &removed)`; C++ and Rust `ext` counterparts. **Pending user approval**                                                                                                                                                                         | The registry otherwise keeps a C# function pointer past a domain reload                                                                                                                                     |

## C ABI additions for v0.2

C# needs no C ABI change for parity with Python: the spike bound ABI 5 from
C# on the tested path (spaces, env vtable, env server, Python driving a
C#-served env end to end), and every area maps its rows onto existing ABI 5
calls. Five additive changes are needed for the C#-only parts that engines
depend on (cancellable `Async` twins, serve twins without process-wide
signal handlers, a shutdown barrier that leaves no managed pointer in native
hands) and for factory serves to report their own runtime. ABI 5 is
unreleased, so each lands in the ABI 5 generation with no bump; struct
fields are appended under R-19, after the ABI-5 `sizeof`, as
`RLMeshRemoteEnvOptions.api_key` was. All five are **pending user
approval** ([table](#c-decisions-pending-user-review)); C++ gains each one
in the same release, and Rust gains what it lacks (X-16 only). A rejected
addition drops the C# feature that needs it from v0.2 (R-16), not its
safety.

### X-12 Operation cancel scope

```c
/* Fenced (R-25). Makes token the calling thread's operation token (a new
 * reference) until rlmesh_ext_cancel_scope_exit. */
RLMESH_API RLMeshStatus rlmesh_ext_cancel_scope_enter(RLMeshExtCancelToken* token);
/* Releases the scope's reference; no-op when none is entered. */
RLMESH_API void rlmesh_ext_cancel_scope_exit(void);
```

- **Contract** (R-18 "Operation cancel scope"). `enter` refuses a NULL
  token with `RLMESH_ERR_INVALID_ARGUMENT` and a second `enter` before
  `exit` at the same callback depth with `RLMESH_ERR_RUNTIME` (no
  nesting); it consumes no firing. While the scope is entered, every call
  that R-18 already makes cancellable, made directly on that thread, also
  watches the scope token: the calls with a `cancel` options field
  (`rlmesh_run`, `rlmesh_model_run`, `rlmesh_session_run`, the session
  constructors, `rlmesh_env_factory_serve`, `rlmesh_ext_env_factory_start`
  until it returns, `rlmesh_model_serve`) and the calls on a handle with
  `cancel_handle()` (`rlmesh_session_*`, `rlmesh_remote_env_*`,
  `rlmesh_remote_vector_env_*`), at exactly the points where they watch
  their own tokens (a dial stays uncancellable, RMT-6). The call
  subscribes the scope token when it starts, before its busy and overlap
  checks, and releases it when it returns; a firing made while nothing
  watched the token fails the next such call at its start and is consumed,
  as R-18's idle `cancel()`; a firing during the call cancels it with R-18's
  semantics, whichever of its tokens fired. Every other call ignores the
  scope. Before invoking a callback the library saves and clears the
  thread's scope and restores it afterwards, as it does R-12's pending
  error, so calls a callback makes are outside the caller's scope; a scope
  a callback enters and leaves entered is exited when the callback returns,
  and a thread that exits with one entered releases its reference.
- **Why a scope, not new functions or fields.** Every single-call twin
  (`Session` reset, step, predict, contract and the other connecting reads;
  the remote clients' reset, step, render, close, shutdown) needs its own
  token, and most have fixed signatures with no options struct: a
  cancellable `rlmesh_ext_` variant of each would double about 25 exports
  and every later one, and a shared-token disarm would erase legitimate
  firings. A thread scope covers them all with two calls and keeps R-18's
  token, firing and state rules unchanged. The facade already composes
  several tokens for one call (the model serve watches its `cancel` and a
  drop token together, `rs/model/facade.rs`).
- **Sizing (R-19), conventions (R-20).** No struct and no `user_data`.
- **Consumers.** C# (every twin's operation, the long-running sync calls,
  the shutdown barrier; R-16, R-18, R-28); C++ `class
rlmesh::ext::CancelScope { public: explicit CancelScope(const
CancelToken&); ~CancelScope(); }` (non-copyable, non-movable; throws the
  class of a refused `enter`); C. Rust has no export: it cancels by dropping
  futures and through `cancel` fields.
- **Tests** (E-3 Cancellation, in C, C++ and C#): a scope token fired
  mid-call cancels `rlmesh_session_step` and leaves the session usable; the
  completion race (fire after the call returned: the next call on the
  handle succeeds) and the admission race (a second thread's overlapping
  call is refused, its scope token fired, the first call completes
  normally), both driven by a scripted env blocking on a barrier; a token
  fired before the call fails it at its start; the scope token and a
  `cancel` field token each cancel a run when fired alone; a hook's own
  call inside a scoped run is not cancelled by the scope; a nested `enter`
  is refused; `exit` with none is a no-op.

### X-13 Background factory serve

```c
/* Fenced (R-25; Python's EnvFactory has no start). Steps 0-5 of
 * EnvFactory.serve, then rlmesh_env_server_start. */
RLMESH_API RLMeshStatus rlmesh_ext_env_factory_start(
    RLMeshEnvFactory* factory, const char* address,
    const RLMeshEnvFactoryServeOptions* options /* NULL = INIT */, RLMeshEnvServer** out);
```

- **Contract.** Runs steps 0 to 5 of Env's `EnvFactory.serve` on the
  calling thread (the factory's callbacks run there, exclusively, ENV-34):
  the pre-fired `cancel` check, the branch check, `prepare`, `num_envs`
  `make_env` calls, the edition, the server with `tags` per ENV-18, the
  describe envelope published on the handshake (ENV-13) with X-14's host
  runtime, and the `RLMesh serving env on …` line. It then starts the
  server as `rlmesh_env_server_start` does (every lane on library threads,
  no signal handlers, ENV-11) and returns it in `*out`, owned by the
  caller. The server owns the envs it made and closes them on shutdown
  (`close_env_on_shutdown` true, as factory serve); it holds nothing of
  the factory, which the caller may use or free at once (never closed, as
  Python's `serve` never closes it). `rlmesh_env_server_wait`, `_shutdown`,
  `_address`, `_env_contract`, `_free` and
  `rlmesh_ext_env_server_describe_json` apply as to any server. `cancel` is
  borrowed for this call only: a token already fired fails before the
  branch check; one fired during steps 1-5 fails the call with the
  cancellation class once the server is built, after shutting it down and
  closing its envs. On failure `*out` is NULL and every env made so far is
  closed. The name takes `rlmesh_ext_` because Python's `EnvFactory` has
  no start (R-25); the user's working name `rlmesh_env_factory_start`
  would put a language-only extra on the shared surface.
- **Implementation note.** The facade already has the pieces: the private
  `prepare_serve` that `serve_host` runs before `EnvServer::serve`
  (`rs/env/factory.rs`) plus `EnvServer::start`; the capi reaches them
  through a `binding` entry point beside `env_factory_native_serve`
  (`rs/binding.rs`).
- **Sizing (R-19).** Reuses `RLMeshEnvFactoryServeOptions` (with X-14).
- **Consumers.** C# `public static EnvServer
Ext.EnvFactoryExtensions.Start(this EnvFactory factory, string address,
EnvFactoryServeOptions? options = null)` and `EnvFactory.ServeAsync` (Start
  plus `WaitAsync`, ENV-39); C++ `EnvServer rlmesh::ext::start(EnvFactoryBase&
factory, std::string_view address, const EnvFactoryServeOptions& = {})`;
  C. Rust has `EnvFactoryExt::serve_async` and gains nothing.
- **Tests** (E-3 Serving): a Python client drives a started factory server
  and reads the same describe envelope a foreground `rlmesh_env_factory_serve`
  publishes; SIGINT and SIGTERM dispositions are unchanged while it serves
  (`sigaction` before and after); a pre-fired `cancel` runs no `prepare`;
  freeing the factory right after the call leaves the server serving;
  `rlmesh_env_server_shutdown` closes every made env once; C++ and C# run
  the same through their wrappers.

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
  read `ext->host_runtime_json` with the same validator as
  `RLMeshExtEnvServerOptions.host_runtime_json` (`capi/describe.rs`
  `host_runtime`: a JSON object with `language`, `language_version`,
  `framework_versions` and a `toolchain` of gcc, clang or msvc); an invalid
  value fails with `RLMESH_ERR_INVALID_ARGUMENT` before the branch check.
  The value is the handshake's PeerInfo and the published envelope's
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
  servers (ENV-12). C#: `EnvFactoryServeOptions.Ext` of type
  `Ext.EnvFactoryServeOptions { string? HostRuntimeJson }`, `null` = the
  wrapper's detected runtime (ENV-38); the Unity integration sets its
  engine's. C. Rust passes its own runtime and has no field.
- **Tests** (E-3 Serving): a C factory serve with NULL `ext` reports
  `"c"`, a C++ one `"c++"`, a C# one `"csharp"`; an invalid JSON value is
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
  the calling thread and stops on its `cancel` token, an operation scope
  (X-12), the idle timeout, a remote shutdown or `rlmesh_model_close`. The
  facade already has this switch: `serve_blocking`'s `foreground` flag,
  false for `Model::serve_async` (`rs/model/facade.rs`). False, the zero
  value, keeps today's behaviour (R-19: zero is the default).
- **Why a field, not `rlmesh_model_start`.** No language has a model server
  handle: Python's `Model.serve` is foreground only and Rust's twin is a
  future. A start call would invent a handle type (address, wait,
  shutdown, free) with no Python counterpart for R-1 to map, while the
  field maps onto an existing facade switch, and a blocking serve on a C#
  pool thread with an operation token already gives `Model.ServeAsync`
  Rust's `serve_async` semantics (no handlers, stops on cancel).
- **Sizing (R-19).** The struct's `sizeof` is a multiple of `sizeof(size_t)`
  with no tail padding, so the field starts at the ABI-5 `sizeof`, which
  stays the minimum; an ABI-5 caller reads as false.
- **Consumers.** C# `Model.ServeAsync` (always set) and
  `Ext.ModelServeOptions.NoSignalHandlers` for a sync `Serve` from an
  engine's own thread (MDL-24); C++ `ext::ModelServeOptions::no_signal_handlers
= false`; C. Rust has `serve_async` and gains nothing.
- **Tests** (E-3 Serving): with the field set, SIGINT / SIGTERM
  dispositions are unchanged while serving and a `cancel` stops the serve
  with the cancellation class; unset, the existing signal tests still pass;
  an ABI-5-sized struct reads as unset.

### X-16 Entrypoint unregistration

```c
/* Fenced (R-25). Drops the registry's reference to entrypoint's transform;
 * routes already resolved keep theirs. */
RLMESH_API RLMeshStatus rlmesh_ext_unregister_entrypoint(const char* entrypoint, bool* out_removed);
```

- **Contract.** Removes `entrypoint` from the process-wide registry that
  `rlmesh_ext_register_entrypoint` fills (MDL-11) and releases the
  registry's `RLMeshObsTransform` reference; the transform's
  `free_user_data` runs once its last reference (a resolved route of a live
  model or adapter) is released. `*out_removed` is false when the name was
  not registered; NULL `entrypoint` or `out_removed` is
  `RLMESH_ERR_INVALID_ARGUMENT`. Today a registration can only be replaced,
  never removed, so a C# transform's function pointer would stay in the
  registry past a domain reload (R-28's shutdown barrier, D-43).
- **Sizing (R-19), conventions (R-20).** No struct; no new `user_data`.
- **Consumers.** C# `Ext.Entrypoints.Unregister(string name) → bool`, and
  the shutdown barrier for every name C# registered; C++ `bool
rlmesh::ext::unregister_entrypoint(std::string_view)` (a plugin that
  unloads); Rust `ext::unregister_entrypoint(name: &str) -> Result<bool>`,
  the facade function the export wraps; C.
- **Tests:** register, resolve with `trust_entrypoints`, unregister: a new
  resolve fails with `AdapterResolutionError` as for an unknown name
  (ADP-11), the existing route keeps working, and `free_user_data` runs
  exactly once, after the route's model is freed; unregistering an unknown
  name returns false.

The generator does not translate `_INIT` macros, `RLMESH_RANDOM_SAMPLE`'s
compound literal or `static inline` helpers (`rlmesh_abi_check` among them);
the hand layer supplies managed equivalents, and E-3 checks them field by
field against C (R-19, R-28). That needs no ABI change.
