# Versioning

RLMesh follows [Semantic Versioning](https://semver.org). The project is pre-1.0, so the SemVer pre-release rule applies: a minor release (`0.x`) may contain breaking changes. We treat that seriously. Every breaking change to a stable surface is called out in the {doc}`changelog` under a Breaking heading, with a migration note.

## What a version bump means

While RLMesh is `0.y.z`:

- **Minor (`0.1` → `0.2`)** may change or remove a stable Python API. Breaking changes ship here, never in a patch.
- **Patch (`0.1.0` → `0.1.1`)** is bug fixes and additive changes that keep the stable surface working.

After `1.0`, RLMesh follows standard SemVer, with breaking changes only in a major release.

## What the contract covers

The version contract applies to the **Python package** (`rlmesh` on PyPI), and only to symbols labeled **Stable**. See {doc}`compatibility` for the full stability surface and what each label promises.

- **Stable** is the surface we intend to keep and will change carefully, with a migration note. It is not frozen until 1.0.
- **Experimental** may change or disappear at any time, without a migration note.

## The Rust crates are internal

RLMesh publishes its Rust crates to crates.io so the Python extension can build. Most of them are internal implementation detail with no stability promise: their Rust API may change at any time and there is no plan to stabilize it. The exceptions are the `rlmesh` facade crate and the CLI commands — the Rust-side surfaces we intend to stabilize. Stabilizing the facade API is an explicit near-term goal, planned but not yet committed; see {doc}`compatibility` for the roadmap. Until it lands, build on the Python package.

## How changes ship

Every release records its user-facing changes in the {doc}`changelog`. A breaking change to a stable symbol is listed under a Breaking heading with a before-and-after migration note. The forward-compatibility details live in {doc}`compatibility`.
