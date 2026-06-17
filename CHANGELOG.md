# Changelog

All notable RLMesh changes are tracked here. Entries are generated from conventional commit messages
with `mise run changelog:build`.

## Unreleased

### Removed

- **Breaking:** recipes: Remove recipe authoring (`rlmesh.recipes`, `register`, `EnvRecipe`,
  `ModelRecipe`) from the package

## 0.1.0-beta.3 - 2026-06-15

### Added

- **Breaking:** spaces: DLPack-native Tensor with zero-copy framework backends (#3)
- **Breaking:** handshake: Negotiated workflow editions, edition spec docs, and generated changelog
  (#2)
- **Breaking:** beta: Harden RLMesh APIs, spaces, and transport (#5)
- recipes: Three-phase environment Recipe system (#8)
- **Breaking:** vector-lifecycle: Per-lane NEXT_STEP autoreset contract for vector environments (#7)
- adapters: Tag-driven IO adapters — env tags × model specs resolved at runtime (#9)
- models: Add ModelRecipe authoring and containerized model eval (#11)

### Fixed

- grpc: Keep tracing spans across awaits and demote per-step logs
- cli: Align CLI tagline with README positioning
- Allow to build again

### Changed

- Final workspace cleanup with workflow-edition and handshake hardening (#10)

## 0.1.0-beta.2 - 2026-06-08

### Added

- docs: Created simple python docs

### Fixed

- sandbox: Fix nested and vector sandboxes

## 0.1.0-beta1

Initial OSS beta release.

- Python SDK and native extension package.
- Rust SDK, protocol, runtime, sandbox, and CLI crates.
- Installed-wheel validation harness for basic and optional heavy profiles.
- Public API snapshot tests for the Python package.
