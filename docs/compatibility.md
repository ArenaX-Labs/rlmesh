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

> [!WARNING]
> The dtype values `int8/16` and `uint16/32/64` are not negotiated. A peer from an earlier release fails with a decode error naming the unknown dtype when it meets an environment that uses them: a clean refusal at the decoding peer, not a conversion. A per-leg dtype ceiling that refuses on the sending side instead is on the roadmap below.

> [!WARNING]
> The `rlmesh-wire-v1` protocol generation stabilized at 0.1.0, and the supported-generation window holds that single generation. A future incompatible wire change mints a new generation rather than mutating v1, and it would live only in the runtime (see [The runtime is the interpreter](#the-runtime-is-the-interpreter)). Prerelease and local builds carry exact cohort suffixes, so mismatched moving builds fail loudly instead of guessing they are compatible.

## Native APIs: Rust, C and C++

The `rlmesh` Rust facade crate, the C API (`rlmesh.h`) and the C++ header (`rlmesh.hpp`) share the Python package's nouns and constructs (envs, spaces, values, models, runs, sessions, remote clients and the same error classes and texts), each in its own language's conventions. Their changes are recorded in the [changelog](../CHANGELOG.md), breaking ones with a migration note. None of them carries Python's Stable promise yet:

- **Rust facade (`rlmesh` crate): Preview.** It is intended to become stable but may still change in a `0.x` minor release, with a migration note. `rlmesh::ext` holds documented Rust-only extras and is part of the facade. `rlmesh::binding` is `#[doc(hidden)]` plumbing for the in-repo bindings (the Python extension and the C API): it is Internal and exempt from SemVer, so do not depend on it.
- **C API (`rlmesh.h`): Experimental.** Binary compatibility is tracked by the ABI generation, currently 6: `RLMESH_ABI_VERSION`, the `librlmesh_capi.so.6` SONAME and the CMake package version. A layout or enum change, a signature retype or a removed symbol bumps the generation, which may happen in any minor release; binaries and sources built for another generation do not load or build against it. Within a generation, a sized struct (options, vtables, attributes, out-structs) grows only by appending optional fields whose all-zero bytes mean the default, so a caller compiled against an older header keeps working; a field that must be supplied needs a new generation. A struct below its `RLMESH_<NAME>_MIN_SIZE` is refused, and a newer caller's nonzero bytes past the library's struct size are refused rather than ignored. C-only extras are fenced under `rlmesh_ext_` / `RLMeshExt` names.
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

Every participant brings two things to its handshake. `supported_workflow_editions` is what it **can** run: the retained list of editions its build implements. `preferred_workflow_edition` is what it **wants**: the one edition it declares. The runtime is the only participant that sees every other one, so it alone selects, and the rule is the highest edition every participant can run that no participant's declaration excludes (`negotiate_session_floor` in `rlmesh-proto`). A participant that declares nothing is read as wanting the newest edition it can run, which is what every build made before the field existed means, so the rule is a no-op against a 0.1.0 peer. A declaration is a ceiling, not an exact demand: a bare `YYYY.MM` admits every cohort of that base and anything older, so `2026.06` works unchanged on a dev or prerelease build whose only offer is `2026.06-<cohort>`, and a declaration above what another participant can run never lifts the session past that participant. A declaration with a cohort suffix uses the full edition ordering as its ceiling, so it can select an older sealed edition both sides offer. The runtime is itself a participant, so it can hold a session below what env and model could run together; when it does, it logs whether its build or its own declaration was the cause. When no edition satisfies everyone, the session is refused before any Join stream opens, with a message naming what each tier (env, model, runtime) wants and can run. The chosen edition is then pinned on both legs: `ResolveAdapterRequest.selected_workflow_edition` to the model, and `ConfigureEnvRequest.selected_workflow_edition` as the env's first Join message.

### The runtime is the interpreter

Env and model never talk to each other; each talks only to the runtime, which decodes and rebuilds every message it relays. So one edition governs the whole session, and the runtime reads every edition-governed default (the step bound, the reserved reset option, which autoreset modes it owns, the success info keys, the conformance-warning key) from a per-edition table keyed by that one value, never from a string compare. Ceilings, by contrast, are per leg: what the runtime may emit toward the env is bounded by what the env's build can decode, and likewise for the model, so a type one leg cannot express is refused on that leg alone. The runtime records those ceilings on the session spec and checks its contract and relayed payloads through `RelayPolicy`; the default policy refuses payloads outside the target ceiling.

The same shape fixes what a future `rlmesh-wire-v2` would look like: a runtime-only dual stack. The runtime registers both generations and speaks v1 to a v1 leg and v2 to a v2 leg. A served env or model never needs both, and a generation bump never partitions a fleet.

### What is part of the contract

- **The 256 MiB message cap.** One encoded message on any env or model leg, in either direction, is bounded by `rlmesh_grpc::MAX_MESSAGE_SIZE`; an oversized message fails at the sender's encode or the receiver's decode with the gRPC status `OUT_OF_RANGE`. It is a fixed constant of `rlmesh-wire-v1`, not a knob, and no later release lowers it (details in [performance](https://rlmesh.dev/docs/guides/performance/)).
- **Sealed editions are never dropped.** `rlmesh.toml` lists the retained editions, the `rlmesh-proto` crate ships a copy of that list, and every build generates its offer from it at build time. `mise run policy:check` fails if the generated list and the manifest disagree, if a sealed edition leaves the list, or if the list lost an edition the last release tag's manifest had sealed.
- **The wire grows additively.** Field tags are never removed or renumbered within `rlmesh-wire-v1` (`mise run protocol:breaking`). A new dtype, `AutoresetMode`, or `SpaceSpec` arm is a new type an old peer refuses rather than decodes wrongly (the two exceptions, error codes and `MetaValue` kinds, are listed under "Not yet guaranteed"), and an emitter may only send it toward a leg that can decode it; the emitter-side rules are in [workflow editions](editions/index.md).

### The machine proof

`mise run test:crossver` builds a release-cohort wheel from this tree and runs it against the published wheel pinned in `tests/system/crossver.lock`: old and new env servers against old and new runtimes, old and new served models against the other side's runtime, both same-version controls, and two forged refusals (a pin of an edition no build implements, and a `rlmesh-wire-v2` handshake). Each real cell must reproduce the committed trace and report the server's handshake result and the session edition. It runs as its own CI job; `docs/testing.md` describes the cells.

### Not yet guaranteed

- Only one edition exists. Negotiation, pinning, and the defaults table are exercised, but no session has run at an edition other than the participants' newest, so edition-driven behavior divergence is untested until a second edition is minted.
- The OSS runtime refuses; it does not convert. A dtype, enum value, or space arm the target leg cannot decode is a decode error at that peer today, not a re-encoding on its behalf. Two unknown-value paths still fold instead of refusing: an unknown env or model error code reads as `UNSPECIFIED`, and a `MetaValue` with an unknown arm reads as `null`, so an emitter must not put new semantics on either.
- Per-leg ceilings check known dtypes and payload sizes. They do not preserve unknown protobuf fields through decoding, and the message transport still enforces the full encoded-message limit.
- The wire-v1 and sealed-edition contracts apply from v0.1.0. Before the first stable wheel is published, mixed cells verify refusal of the incompatible rc.12 cohort. The permanent fixture becomes v0.1.0 after publication.

## Versioning and forward-compatibility roadmap

The wire and behavioral contracts are sealed at v0.1.0. Further work extends the implementation and its evidence:

- **v0.1.0.** First stable release. Seals the `2026.06` workflow edition, freezing its spec checksum, and freezes the `rlmesh-wire-v1` protocol generation.
- **Included in v0.1.0.** A typed edition and per-edition defaults table in the runtime; WANT/CAN negotiation over `preferred_workflow_edition`; declaration surfaces with a base-level ceiling; the `ConfigureEnv` pin on the env leg; the retained edition list generated from `rlmesh.toml` and gated by `policy:check`; the cross-version matrix in CI; `rlmesh.build_info()`; a reserved `MetaValue.null` arm.
- **Second edition, when a real semantic change requires one.** Mint a second workflow edition and exercise negotiation and the defaults table against a real semantic change.
- **Included in v0.1.0: per-leg ceilings and the relay seam.** The runtime records each leg's decodable dtypes, capabilities, and message cap. Its relay policy checks the contract and payloads before forwarding them. Conversion for an older leg is a policy the managed platform may supply; the OSS default refuses.
- **`rlmesh init`.** A scaffolder that writes the edition declaration into a new project, so declaring is the default rather than a step to remember.
- **Native APIs, near term.** Promote the `rlmesh` facade crate from Preview to Stable and stabilize the CLI commands once they settle. The other crates stay internal with no stability promise.
- **v1.0, date not set.** Broader API stabilization. The wire-v1 and sealed-edition commitments already apply from v0.1.0.

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

`mise run check` includes `mise run policy:check`, `mise run protocol:baseline-verify` (live protos byte-identical to the committed snapshot), and `mise run protocol:breaking` (`buf breaking` against the immutable `v0.1.0` tag). Before that tag exists, the v0.1.0 release candidate is checked against the rc.12 tag. Later versions require the stable tag. Regenerating the editable snapshot cannot make an incompatible change pass this comparison. The byte goldens in `crates/rlmesh-proto/tests/wire_golden.rs` also pin each message's encoding.
