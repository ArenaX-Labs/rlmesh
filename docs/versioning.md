# Versioning

RLMesh follows [Semantic Versioning](https://semver.org). The project is pre-1.0, so SemVer’s initial-development rule applies: a minor release (`0.x`) may contain breaking changes. Every breaking change to a stable surface is called out in the [changelog](../CHANGELOG.md) under a Breaking heading, with a migration note.

## What a version bump means

While RLMesh is `0.y.z`:

- **Minor (`0.1` → `0.2`)** may change or remove a stable Python API. Breaking changes ship here, never in a patch.
- **Patch (`0.1.0` → `0.1.1`)** is bug fixes and additive changes that keep the stable surface working.

After `1.0`, RLMesh follows standard SemVer, with breaking changes only in a major release.

## What the contract covers

The version contract applies to the **Python package** (`rlmesh` on PyPI), and only to symbols labeled **Stable**. See [compatibility](compatibility.md) for the full stability surface and what each label promises.

- **Stable** is the surface we intend to keep and will change carefully, with a migration note. It is not frozen until 1.0.
- **Preview** is intended to become stable but may still change in a minor release, with a migration note. Today it labels only the `rlmesh` Rust facade crate.
- **Experimental** may change or disappear at any time, without a migration note.

## Rust, C and C++

RLMesh publishes its Rust crates to crates.io so the Python extension can build. Most of them are internal implementation detail with no stability promise: their Rust API may change at any time and there is no plan to stabilize it. The `rlmesh` facade crate is Preview; the C API (`rlmesh.h`) and the C++ header (`rlmesh.hpp`) are Experimental, with binary compatibility tracked by the C ABI generation rather than the package version. The CLI commands are the other Rust-side surface we intend to stabilize. [Compatibility](compatibility.md#native-apis-rust-c-and-c) has the details and the roadmap; for the strongest promise, build on the Python package.

0.2 publishes only Python wheels and Rust crates. C/C++, C#, Unity, Unreal and
npm remain in the tree with CI coverage, but publication is fenced. C ABI 7 is
cut when the native bindings ship. See [release scope](release.md#github-releases).

## Workflow editions and the version contract

The package version and the workflow edition answer different questions. The version says which Python API a release ships and follows SemVer above; the edition says what a session means, is declared once in an env's or model's source, and stays put across upgrades. A minor release may mint a new edition when a real semantic change requires one (at most one per release). Every release of a line keeps offering the editions sealed in that line, so a declared participant keeps running at its edition across patch upgrades. Before 1.0 a new minor may also stop offering an older edition or replace the wire (0.2 did both), so a minor upgrade can need a rebuild; keeping editions across every release is the 1.0 goal. The guarantee and its current limits are in [compatibility](compatibility.md); declaring an edition is in [workflow editions](editions/index.md).

Frozen container images reproduce their results on their matching runner. An
edition locks defaults within its release line; it does not make a 0.1 image
compatible with 0.2. Interoperation between the 0.1 and 0.2 protocols is not implemented.

## How changes ship

Every release records its user-facing changes in the [changelog](../CHANGELOG.md). A breaking change to a stable symbol is listed under a Breaking heading with a before-and-after migration note. The forward-compatibility details live in [compatibility](compatibility.md).
