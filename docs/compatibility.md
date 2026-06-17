# Compatibility

RLMesh documents compatibility at the workflow level rather than treating every internal type as
frozen. The project is released and pre-1.0: stability labels describe the support level today, and
the surface still evolves with care until 1.0. See {doc}`versioning` for the version contract.

```{note}
**This documents 0.1.0-rc.1, a release candidate.** The `2026.06` workflow edition is still
provisional; the stable cutover — sealing the edition and the `Production/Stable` trove status —
lands at the final 0.1.0. The stability descriptions below describe that release. Install the
candidate with `pip install rlmesh==0.1.0rc1`.
```

```{note}
RLMesh is pre-1.0 (`0.x`). "Stable" means the surface we intend to keep and will change carefully, with a migration note in the {doc}`changelog` — not an API frozen until 1.0. "Experimental" may change or disappear. A `0.x` minor release may break a stable API, so pin a minor range for active projects.
```

## Stable

Stable workflows include documented public APIs, supported CLI flows, and supported remote
environment/model interactions. Stable means the surface we intend to keep and will change
carefully, with migration notes — not a frozen API.

- Imports, signatures, and documented behavior follow the version contract: a breaking change to a
  stable symbol ships in a minor release with a migration note in the {doc}`changelog`.
- Peers must currently run the same release. Cross-version acceptance, where newer runtimes keep
  accepting older stable clients and packages, is on the roadmap below, not a guarantee today.
- New features may require newer packages or capabilities, but older stable workflows should fail
  clearly or keep working.

## Preview and Experimental

Preview APIs are intended to become stable but may still change with migration notes. Experimental
APIs may change or disappear. Preview is reserved for the intended-stable-but-still-moving case and
is currently unused; today's labels are only Stable and Experimental.

Torch and JAX backends and sandbox helpers are experimental. The `MultiBinary`, `MultiDiscrete`,
`Text`, and `Tuple` space wrappers are also experimental; see {doc}`gymnasium` for the per-space
stability labels, which track the API surface policy in `api_metadata.json`.

```{warning}
The dtype values `int8/16`, `uint16/32/64`, and `bfloat16` are not negotiated. A peer from an
earlier release fails with a decode error naming the unknown dtype when it meets an environment
that uses them, so run both ends on the same release. An edition-gated dtype negotiation floor is
on the roadmap below.
```

```{warning}
The `rlmesh.protocol.v1` wire format stabilizes at 0.1.0. The earlier 0.1.0 beta releases are not
wire-compatible with it; rebuild both peers when upgrading from a beta.

The supported-generation window currently holds a single generation, so peers must run the same
release. A future incompatible wire change mints a new generation rather than mutating v1; a
cross-version generation window is on the roadmap below. The `2026.06` workflow edition seals at
0.1.0 — it is provisional in this release candidate — after which its spec checksum is frozen and a
matching edition string is enough for peers to interoperate.
```

## Rust crates

Most Rust crates are internal implementation detail with no stability promise: they are published to
crates.io so the Python extension can build, but their Rust API may change at any time and there is
no plan to stabilize it. The `rlmesh` facade crate and the CLI commands are the exceptions — the
Rust-side surfaces we intend to stabilize. Stabilizing the facade API is a near-term goal (see the
roadmap below); until then, build on the Python package. See {doc}`versioning`.

## Framework Version Floors

The optional framework backends declare the lowest versions their conversion paths actually require.
Each floor has a concrete reason:

| Package | Floor      | Why                                                                 |
| ------- | ---------- | ------------------------------------------------------------------- |
| Python  | `3.10`     | Ecosystem baseline; all framework floors below ship `cp310` wheels. |
| numpy   | `>=1.22`   | First release with complete Python 3.10 wheel coverage.             |
| torch   | `>=1.11`   | First release with full `cp310` wheel coverage. [^torch-glibc]      |
| jax     | `>=0.4.24` | First release with DLPack `bool` support.                           |

[^torch-glibc]:
    Torch wheels older than 1.13 fail to load on glibc 2.41+ hosts ("cannot enable executable
    stack"), so the floor harness exercises 1.13.1 there; 1.11 remains the declared install floor
    for older systems.

The floor harness runs via `mise run test:python:floors`, which builds a `cp310` wheel and runs the
framework test suites against exactly these versions. Versions below a floor may work but are
unsupported. Within a framework, some features need newer releases: `rlmesh.numpy` itself converts
through the buffer protocol on any supported numpy, but consuming RLMesh tensors with
`np.from_dlpack` needs numpy 1.23 (`bool` needs 1.25). Torch `bool` over DLPack needs 2.2 (older
versions fall back to a copy), and Torch `uint16/32/64` need 2.3.

## Value Semantics and Caveats

`rlmesh.Tensor` is a validated transport container with DLPack and buffer-protocol edges. It is not
an ndarray. Compute, slicing, and broadcasting belong to the frameworks; RLMesh moves bytes and
metadata between them and the wire.

- Zero-copy is asymmetric: exporting (`memoryview`, `__dlpack__`, framework views) is zero-copy;
  importing (constructing `Tensor`, `Tensor.from_dlpack`) currently always copies. Zero-copy import
  is planned.
- Integer precision: Box bounds carry dtype-typed bytes for integer/boolean dtypes (a single scalar
  for uniform bounds, one per element otherwise, little-endian in the space's dtype), and
  containment compares in the dtype's native domain, so `int64`/`uint64` bounds and values are exact
  to the full range (including `i64::MIN`, `i64::MAX`, and `u64::MAX`). Float dtypes keep the
  `double`-based bounds. The legacy scalar-list wire encoding still stores integers in a signed
  64-bit slot, so `uint64` values above 2^63 wrap on that path (the raw byte encoding used by modern
  clients is exact).
- Mutation: in-place preprocessing on a decoded observation never corrupts the wire buffer. NumPy
  and Torch decode to owned, writable copies; JAX decodes to an immutable array. The explicit
  zero-copy views (`from_dlpack`, the buffer protocol, `torch.as_tensor(copy=False)`) are read-only;
  NumPy enforces this, Torch does not (see the Torch backend page).

## Workflow Editions

Workflow semantics are governed by a negotiated workflow edition. Each edition string names an
immutable behavioral contract documented in {doc}`editions/index`; the handshake selects the highest
edition supported by both peers. Editions change only on deliberate semantic redesigns; new features
and new APIs do not mint editions. The current edition, `2026.06`, seals at 0.1.0 and is provisional
in this release candidate.

## Versioning and forward-compatibility roadmap

RLMesh commits to forward-compatibility guarantees only once the code enforces them and a
cross-version path is proven. The items below are planned, not promises; each is announced here as
it ships.

- **v0.1.0 (upcoming).** First stable release. It seals the `2026.06` workflow edition, freezing its
  spec checksum. Peers still run the same release. The 0.1.0-rc.1 candidate ships first with the
  edition still provisional.
- **Hardening, around July 2026.** A cross-version test harness and a shared compatibility helper, a
  re-snapshotted protocol baseline with stricter checks, and the workflow edition made load-bearing
  in the runtime. This unlocks edition-driven behavior and a cross-version path once a second
  edition exists.
- **Forward tolerance, around late July 2026.** Edition retention guarantees, a dtype negotiation
  floor, and adapter forward-tolerance.
- **Second edition, around August 2026.** Mint a second workflow edition to exercise negotiation
  against a real semantic change.
- **Rust facade API, near term.** The `rlmesh` facade crate and the CLI commands are the Rust-side
  surfaces we intend to stabilize once they settle; the other crates stay internal with no stability
  promise. The commitment is announced here when it lands.
- **v1.0, date not set.** Forward-compatibility guarantees — newer runtimes accept older stable
  clients, and sealed editions are never pruned — become binding, gated on the hardening above and a
  proven cross-version path.

## Value conformance

The `2026.06` edition defines how observation and action values are checked against their declared
spaces (full contract: {doc}`editions/2026.06`). Two points matter in practice:

- **Out-of-bounds values warn; they do not fail.** A `Box` value outside its bounds, or a `Text`
  value outside its charset or length, is delivered and reported once in the `reset`/`step` info map
  under the `rlmesh.conformance.warning` key. This keeps the many Gymnasium environments whose
  values drift past their declared bounds usable out of the box. Set
  `RLMESH_VALIDATION_POLICY=strict` to reject such values instead, or `off` to skip the checks.
  Structural problems — wrong shape, dtype, arity, or domain, a missing key — and `NaN` are always
  rejected, regardless of the policy.
- **Dtypes are coerced, not passed through.** A value is always converted to its declared dtype
  before transport, so a peer reading the negotiated space never sees a per-message dtype. This is a
  deliberate difference from Gymnasium, which warns but forwards the mismatched dtype (see
  {doc}`gymnasium`). A float supplied for an integer dtype is rejected unless every element is
  exactly integral.

## Artifact Versions

Core feature releases move together. Patch releases may be artifact-specific when the fix is
isolated.

## Enforcement

`rlmesh.toml` records the current package family, artifacts, protocol generation, workflow edition,
and API surface policy:

```bash
python scripts/check_rlmesh_policy.py
```

`mise run check` includes `mise run policy:check`. Pull requests also run protobuf breaking-change
checks with `mise run protocol:breaking` against the checked-in protocol baseline at
`crates/rlmesh-proto/baselines/rlmesh.protocol.v1`.
