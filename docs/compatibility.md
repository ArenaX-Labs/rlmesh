# Compatibility

RLMesh documents compatibility at the workflow level rather than freezing every internal type. The project is released and pre-1.0: the stability labels below describe the support level today. See [versioning](versioning.md) for the version contract.

> [!NOTE]
> RLMesh is pre-1.0 (`0.x`). "Stable" means the surface we intend to keep and will change carefully, with a migration note in the [changelog](../CHANGELOG.md), not an API frozen until 1.0. "Experimental" may change or disappear. A `0.x` minor release may break a stable API, so pin a minor range for active projects.

## Stable

Stable workflows include documented public APIs, supported CLI flows, and supported remote environment/model interactions.

- Imports, signatures, and documented behavior follow the version contract: a breaking change to a stable symbol ships in a minor release with a migration note in the [changelog](../CHANGELOG.md).
- A participant authored against a sealed workflow edition keeps working with later releases of the other participants, and the session runs at its declared edition. What that covers today, and what it does not yet, is under [Workflow Editions](#workflow-editions) below.
- New features may require newer packages or capabilities, but older stable workflows either keep working or fail clearly.

The adapter authoring and resolution surface is Stable. `Adapter.__init__`
and `Adapter.serve_route` are Internal binding plumbing. The
`Adapter.transform_obs_value` and `Adapter.transform_action_value` methods are
Bridge APIs for selecting a `ValueBridge`; application code normally uses
`transform_obs` and `transform_action`.

`sanitize_metadata` is a Bridge API: it coerces Python objects into portable
metadata and has no native-language counterpart.

## Preview and Experimental

Preview APIs are intended to become stable but may still change with migration notes. Experimental APIs may change or disappear. In the Python package today's labels are Stable, Experimental, Bridge (Python-specific conversion helpers) and Internal (binding plumbing); Preview currently applies only to the `rlmesh` Rust facade crate (see [Native APIs](#native-apis-rust-c-and-c)).

Torch and JAX backends and sandbox helpers are experimental. The `MultiBinary`, `MultiDiscrete`, `Text`, and `Tuple` space wrappers are also experimental; see [gymnasium](https://rlmesh.dev/docs/start/gymnasium/) for the per-space stability labels, which track the API surface policy in `api_metadata.json`.

The `params` module and the `Model.params` and `EnvFactory.params` declarations
are Experimental; their native counterparts are deferred until that surface is
Stable.

`rlmesh.EvalReport` (per-task and per-suite success of saved result bundles)
is Experimental. Rust reaches it as `rlmesh::binding::report` (binding
plumbing, semver-exempt) and the CLI as `rlmesh eval report`; C and C++ are
deferred.

`BatchOptions` (a served model's batcher) and the `RunHooks`
`transform_observation` and `transform_action` hooks are Experimental.

`Model.run_routes` (many env targets against one model in one process) and
`RunResult.merge` are Experimental. Rust reaches them as
`rlmesh::binding::routes` (binding plumbing, semver-exempt); C and C++ are
deferred.

> [!WARNING]
> The dtype values `int8/16` and `uint16/32/64` are not negotiated. A peer from an earlier release fails with a decode error naming the unknown dtype when it meets an environment that uses them: a clean refusal at the decoding peer, not a conversion. A per-leg dtype ceiling that refuses on the sending side instead is on the roadmap below.

> [!WARNING]
> RLMesh 0.2 speaks `rlmesh-realtime-v1` over session addresses only. The 0.1 gRPC protocol requires a 0.1 runner. Prerelease and local builds carry exact cohort suffixes so incompatible moving builds fail clearly.

## Native APIs: Rust, C and C++

The `rlmesh` Rust facade crate, the C API (`rlmesh.h`) and the C++ header (`rlmesh.hpp`) share the Python package's nouns and constructs (envs, spaces, values, models, runs, sessions, remote clients and the same error classes and texts), each in its own language's conventions. Their changes are recorded in the [changelog](../CHANGELOG.md), breaking ones with a migration note. None of them carries Python's Stable promise yet:

- **Rust facade (`rlmesh` crate): Preview.** It is intended to become stable but may still change in a `0.x` minor release, with a migration note. `rlmesh::ext` holds documented Rust-only extras and is part of the facade. `rlmesh::binding` is `#[doc(hidden)]` plumbing for the in-repo bindings (the Python extension and the C API): it is Internal and exempt from SemVer, so do not depend on it.
- **C API (`rlmesh.h`): Experimental.** Binary compatibility is tracked by the ABI generation, currently 6: `RLMESH_ABI_VERSION`, the `librlmesh_capi.so.6` SONAME and the CMake package version. 0.2 does not ship the C ABI; generation 6 grew after it was cut without a bump, so the first release that ships it publishes that surface as ABI 7. A layout or enum change, a signature retype or a removed symbol bumps the generation, which may happen in any minor release; binaries and sources built for another generation do not load or build against it. Within a generation, a sized struct (options, vtables, attributes, out-structs) grows only by appending optional fields whose all-zero bytes mean the default, so a caller compiled against an older header keeps working; a field that must be supplied needs a new generation. A struct below its `RLMESH_<NAME>_MIN_SIZE` is refused, and a newer caller's nonzero bytes past the library's struct size are refused rather than ignored. C-only extras are fenced under `rlmesh_ext_` / `RLMeshExt` names.
- **C++ header (`rlmesh.hpp`): Experimental.** A header-only C++17 wrapper over the C API, compiled into each consumer; it changes together with the C API it wraps. It requires C++17 and exceptions and is tested at C++17 and C++20 with one header shape. C++-only extras live in `rlmesh::ext`.

- **C# package (`RLMesh`): Experimental.** The `netstandard2.1` and `net10.0` assemblies support serving authored environments, including spaces, tensors, values and supporting errors. See [the C# guide](../csharp/README.md) for usage and native-library requirements. This API may change or disappear. Hosts must complete `RLMesh.Ext.Lifetime.Shutdown` before unloading its managed context; a `false` result means the context must remain loaded. Unity runtime support has not yet been validated.

The other Rust crates are internal implementation detail with no stability promise: they are published to crates.io so the Python extension can build, but their Rust API may change at any time and there is no plan to stabilize it. The CLI commands are the other Rust-side surface we intend to stabilize. Every crate declares a `rust-version` matching the toolchain pinned in `mise.toml`, which CI builds and tests with; it moves up with that toolchain. See [versioning](versioning.md).

## Framework Version Floors

The optional framework backends declare the lowest versions their conversion paths actually require. Each floor has a concrete reason:

| Package | Floor      | Why                                                                 |
| ------- | ---------- | ------------------------------------------------------------------- |
| Python  | `3.10`     | Ecosystem baseline; all framework floors below ship `cp310` wheels. |
| numpy   | `>=1.22`   | First release with complete Python 3.10 wheel coverage.             |
| torch   | `>=1.11`   | First release with full `cp310` wheel coverage. [^torch-glibc]      |
| jax     | `>=0.4.24` | First release with DLPack `bool` support.                           |

[^torch-glibc]: Torch wheels older than 1.13 fail to load on glibc 2.41+ hosts ("cannot enable executable stack"), so the floor harness exercises 1.13.1 there; 1.11 remains the declared install floor for older systems.

The floor harness runs via `mise run test:python:floors`, which builds a `cp310` wheel and runs the framework test suites against exactly these versions; it is gated in the weekly `Dependency Floors` CI workflow and in `mise run release:check`, so a release cannot ship with untested floors. Versions below a floor may work but are unsupported. Within a framework, some features need newer releases: `rlmesh.numpy` itself converts through the buffer protocol on any supported numpy, but consuming RLMesh tensors with `np.from_dlpack` needs numpy 1.23 (`bool` needs 1.25). Torch `bool` over DLPack needs 2.2 (older versions fall back to a copy), and Torch `uint16/32/64` need 2.3.

## Value Semantics and Caveats

`rlmesh.Tensor` is a validated transport container with DLPack and buffer-protocol edges. It is not an ndarray. Compute, slicing, and broadcasting belong to the frameworks; RLMesh moves bytes and metadata between them and the wire.

- Zero-copy is asymmetric: exporting (`memoryview`, `__dlpack__`, framework views) is zero-copy; importing (constructing `Tensor`, `Tensor.from_dlpack`) currently always copies. Zero-copy import is planned.
- Integer precision: Box bounds carry dtype-typed bytes for integer/boolean dtypes (a single scalar for uniform bounds, one per element otherwise, little-endian in the space's dtype), and containment compares in the dtype's native domain, so `int64`/`uint64` bounds and values are exact to the full range (including `i64::MIN`, `i64::MAX`, and `u64::MAX`). Float dtypes keep the `double`-based bounds. `Scalar`, the dtype-independent decode view in `rlmesh-spaces`, has no unsigned variant, so a `uint64` element above 2^63 is carried through it as the wrapped `i64` bit pattern and reinterpreted by `Scalar::as_u64`; the wire bytes themselves are exact.
- Mutation: in-place preprocessing on a decoded observation never corrupts the wire buffer. NumPy and Torch decode to owned, writable copies; JAX decodes to an immutable array. The explicit zero-copy views (`from_dlpack`, the buffer protocol, `torch.as_tensor(copy=False)`) are read-only; NumPy enforces this, Torch does not (see the Torch backend page).

## Workflow Editions

Workflow semantics are governed by a negotiated workflow edition. Each base edition names a behavioral contract documented in [workflow editions](editions/index.md); prerelease and local builds append exact cohort suffixes. Editions change only on deliberate semantic redesigns; new features and new APIs do not mint editions. The `2026.06` edition sealed at 0.1.0.

An edition is a sticky declaration in a participant's source, not a version: it records the contract an env or model was authored against, and upgrading the rlmesh package never moves it. How to declare one, and which surface wins when several are set, is in [workflow editions](editions/index.md).

### The guarantee

A participant authored against a sealed edition keeps working with any later participant, as long as the later one does not require something the older one cannot express. The session runs at the older participant's declared edition. A genuinely new type the old side never knew is a clean refusal, not a silent conversion.

### How the session edition is chosen

A served model publishes its declared and supported workflow editions on
Attach and in the roster; the env publishes its offer in env info. The driver
chooses the highest edition env, model and runtime can all run, bounded by their
declarations, and pins the model's ModelResolve. The current 0.2 line offers
only 2026.10, so no separate env pin is needed. See [workflow editions](editions/index.md).

### What is part of the contract

- **Sealed editions are retained within a major line.** Every later release of the line keeps offering a sealed edition; only a new major line (each 0.x minor, before 1.0) may drop one, by name and with its reason in `[workflow.dropped_editions]`. `rlmesh.toml` lists the retained editions, the `rlmesh-proto` crate ships a copy of that list, and every build generates its offer from it at build time. `mise run policy:check` fails if the generated list and the manifest disagree, if a sealed edition leaves the list, or if the list lost an edition the last release tag's manifest had sealed.

### Across 0.1 and 0.2

0.2 speaks only session addresses and refuses the 0.1 gRPC wire by address, naming the 0.2 form, so a 0.1 peer and a 0.2 peer never meet and there is no cross-version matrix. An image built with 0.1 is refused by `SandboxEnv` and `SandboxModel` with a message to rebuild it with 0.2.

### Current limits

The realtime protocol remains provisional until 0.2.0 freezes it. Internal run
driver messages remain protobuf data types for now; they are not served remotely.

## Versioning roadmap

0.2.0 seals the 2026.10 workflow edition and freezes the realtime session
protocol. The published 0.1 crates continue to support the 0.1 wire independently.
Native Rust APIs remain Preview; C, C++ and C# remain Experimental. The fenced
C ABI is not released in 0.2 and will ship as ABI 7 in a later release.

## Value conformance

The `2026.06` edition defines how observation and action values are checked against their declared spaces (full contract: [2026.06](editions/2026.06.md)). Two points matter in practice:

- **Out-of-bounds values warn; they do not fail.** A `Box` value outside its bounds, or a `Text` value outside its charset or length, is delivered and reported once in the `reset`/`step` info map under the `rlmesh.conformance.warning` key. This keeps the many Gymnasium environments whose values drift past their declared bounds usable out of the box. Set `RLMESH_VALIDATION_POLICY=strict` to reject such values instead, or `off` to skip the checks. Structural problems (wrong shape, dtype, arity, or domain, a missing key) and `NaN` are always rejected, regardless of the policy.
- **Dtypes are coerced, not passed through.** A value is always converted to its declared dtype before transport, so a peer reading the negotiated space never sees a per-message dtype. This is a deliberate difference from Gymnasium, which warns but forwards the mismatched dtype (see [gymnasium](https://rlmesh.dev/docs/start/gymnasium/)). A float supplied for an integer dtype is rejected unless every element is exactly integral.

## Artifact Versions

Core feature releases move together. Patch releases may be artifact-specific when the fix is isolated.

## Enforcement

`rlmesh.toml` records the current package family, artifacts, protocol generation, workflow edition, and API surface policy:

```bash
python scripts/check_rlmesh_policy.py
```

`mise run check` includes `policy:check`, `protocol:baseline-verify` and
`protocol:breaking`. The protocol checks govern only entries under
`[protocols.*]`: while provisional, lint and session byte goldens protect the
realtime wire; once frozen, the baseline and immutable freeze tag protect it.
The kept `rlmesh-proto` messages are internal driver data, outside wire governance.
