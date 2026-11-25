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

## The four versions

A build carries four version numbers. Each answers one question:

| Version             | Example              | Answers                                                                                                                                                                    |
| ------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Package version     | `0.2.0`              | Which API a release ships, under SemVer above.                                                                                                                             |
| Protocol generation | `rlmesh-realtime-v1` | Which wire family a build speaks. Two generations never talk.                                                                                                              |
| Wire revision       | `2`                  | Whether two builds of one generation can talk. It is the only compatibility gate inside a generation: builds with the same revision talk, whatever their package versions. |
| Workflow edition    | `2026.10`            | Which row of behavioural defaults a participant declared. It is declared in source and never negotiated; see [workflow editions](editions/index.md).                       |

`rlmesh.build_info()` reports all four.

## Release channels

- **alpha** may break anything, the wire included.
- **beta** freezes the wire for its minor: every later build of that minor speaks the same wire revision.
- **rc** takes fixes only.

## What interoperates

- Before 1.0, every 0.2.x build talks to every 0.2.x build. Nothing is promised across minors: 0.3 may replace the wire.
- Prereleases interoperate whenever their wire revisions match.
- From 1.0, any 1.x runtime runs any image built on 1.0 or later, and every 1.x release keeps every edition it has shipped.

A new edition is created only when a default actually changes, never on a schedule or with a version bump.

Frozen container images reproduce their results on their matching runner. Interoperation between the 0.1 and 0.2 protocols is not implemented.

## How changes ship

Every release records its user-facing changes in the [changelog](../CHANGELOG.md). A breaking change to a stable symbol is listed under a Breaking heading with a before-and-after migration note. The forward-compatibility details live in [compatibility](compatibility.md).
