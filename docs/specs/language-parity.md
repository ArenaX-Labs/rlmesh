# Language parity

Status: Draft — reviewed round 6.

RLMesh is one library in four languages: Python (the reference), Rust (the
`rlmesh` facade crate), C (`crates/rlmesh-capi/include/rlmesh.h`) and C++
(`rlmesh.hpp`, wrapping C one to one). If you know the Python API you know
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
   planned enforcement.
2. Then read the area you work on. Each area maps every Stable Python symbol
   it owns to Rust, C and C++, and lists its own Python changes and current
   differences. A symbol shared by several areas has one owner (rules R-27);
   the others link to it.
3. IDs are stable: `R-n` rules, `D-n` decisions, `P-n` Python changes, `X-n`
   cross-cutting changes, `G-n` gaps, `E-n` enforcement. Area files use
   `<AREA>-n` for decisions and `<AREA>-Pn` for Python changes (`COR`, `SPC`,
   `ENV`, `MDL`, `RUN`, `RMT`, `ADP`). Cite them in code review and commit
   messages.
4. "Default — pending user" marks a choice applied everywhere until the user
   settles it. Every one is listed under [Pending decisions](#pending-decisions).

## Files

| File                                       | Covers                                                                                                                                    |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [rules.md](language-parity/rules.md)       | Rules, decisions, error matrix, warnings, Python changes, shapes, known gaps, enforcement                                                 |
| [core.md](language-parity/core.md)         | `Value`, `Metadata`, error detail fields, workflow editions, describe, build info                                                         |
| [spaces.md](language-parity/spaces.md)     | Spaces, `SpaceSpec`, space values and `Tensor`                                                                                            |
| [env.md](language-parity/env.md)           | Env authoring and serving: `Env`, `VectorEnv`, `EnvFactory`, `Variant`, `EnvServer`, `ServeOptions`, `EnvContract`, `trial_index`         |
| [model.md](language-parity/model.md)       | Model authoring: `Policy`, `Model`, `PredictContext`, `SpecArg`, `NO_ADAPTER`                                                             |
| [running.md](language-parity/running.md)   | `run`, `session`, `Session`, run and env targets, `RunContext`, `RunHooks`, results, `View`, `Recorder`, `RANDOM_SAMPLE`                  |
| [remote.md](language-parity/remote.md)     | `RemoteEnv`, `RemoteVectorEnv`, `RemoteModel`, their options, sandbox clients                                                             |
| [adapters.md](language-parity/adapters.md) | `EnvTags`, `ModelSpec`, `ObservationRoles`, transforms, `tag`, `resolve`, `Adapter`                                                       |

## Pending decisions

Each is applied everywhere as stated in its row until the user answers. The
default is the spec as written; a "no" answer takes the alternative named in
the decision.

| ID     | Question                                                                                                                         | Default                                     |
| ------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| D-9    | Is `sanitize_metadata` relabelled Bridge, with no native counterpart?                                                           | Yes (P-7)                                   |
| COR-2  | Does `Metadata` keep insertion order like a Python `dict`, rather than sorting keys?                                             | Insertion order                             |
| COR-4  | Is `is_recoverable` false for every class but Environment and Model (no transport retry flag)?                                  | Yes                                         |
| COR-14 | Keep `kind` on native describe as a cross-check, and make Python refuse a contradicting `kind`?                                  | Yes (COR-P1)                                |
| SPC-6  | Does `Box.contains` check the shape exactly?                                                                                     | Yes (SPC-P7)                                |
| SPC-7  | Are non-integral bounds on integer or bool Box dtypes refused instead of truncated?                                              | Refused (SPC-P6)                            |
| SPC-8  | Does a typed-uniform Box read `low` / `high` back as one number?                                                                 | Yes (SPC-P5)                                |
| SPC-18 | Keep identity equality for `SpaceSpec` and `Tensor` (no native `==` except Rust `Tensor`), deferring structural `__eq__`?        | Yes                                         |
| SPC-21 | Does `RANDOM_SAMPLE` seed its sampler with each episode's reset seed?                                                            | Yes (SPC-P8)                                |
| SPC-22 | Defer per-element Box construction to fenced `ext` builders?                                                                     | Yes                                         |
| SPC-23 | Keep Experimental space constructors, deferring their attribute accessors?                                                       | Yes                                         |
| ENV-13 | Does only `EnvFactory.serve` publish the describe envelope, not a bare `EnvServer`?                                              | Yes; C keeps it via `ext.publish_describe`  |
| ENV-15 | Do native factories declare `TagParam { name, default_, choices }` for the branch table?                                         | Yes                                         |
| ENV-23 | Serve `!Send` Rust envs through a fenced foreground-only `ext::LocalEnvServer`?                                                  | Yes                                         |
| MDL-7  | Serialize corner and hook calls on one Model across all sessions, Python's local sessions included?                              | Yes (MDL-P3)                                |
| MDL-8  | Drop env-tagged episode entries silently on evict-all and release?                                                               | Yes (MDL-P4)                                |
| MDL-11 | Keep `trust_entrypoints` natively, backed by a fenced entrypoint registry?                                                       | Yes (ADP-11)                                |
| RUN-4  | Keep `Model.run` without a viewer (a set `view` with a local `Model` is TypeError)?                                              | Yes                                         |
| RUN-17 | Promote `FrameSources` and the recorder records to Stable in Python?                                                             | Yes (RUN-P5)                                |
| RMT-2  | Keep the client `address` in the options struct rather than adding it to R-7 as a positional?                                   | Options field                               |
| RMT-4  | Does a zero client timeout mean "no timeout", as Python reads it?                                                                | Yes                                         |
| RMT-5  | Keep endpoint tokens off the shared surface, in fenced `ext` constructors?                                                       | Yes                                         |
| RMT-13 | Does I/O on a closed remote client raise `RuntimeError` instead of `ConnectionError`?                                            | `RuntimeError` (RMT-P2)                     |
| ADP-15 | Give unexported Literal types native names derived from their annotation?                                                        | Yes                                         |
| ADP-16 | Relabel `Adapter.__init__` / `serve_route` Internal and the `*_value` transforms Bridge?                                          | Yes (ADP-P4)                                |
| ADP-17 | Keep the core resolution code in `binding`, exposing only the message on `AdapterResolutionError`?                               | Yes                                         |
