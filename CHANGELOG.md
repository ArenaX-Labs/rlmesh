# Changelog

All notable RLMesh changes are tracked here. Entries are generated from conventional commit messages
with `mise run changelog:build`.

## Unreleased

### Added

- **Breaking:** proto: Negotiate workflow editions in handshake
- **Breaking:** handshake: Negotiate workflow editions across servers and clients

### Fixed

- grpc: Keep tracing spans across awaits and demote per-step logs
- cli: Align CLI tagline with README positioning

### Documentation

- editions: Add workflow edition model and 2026.06 spec

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
