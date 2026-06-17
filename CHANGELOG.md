# Changelog

All notable changes to RLMesh are documented here. This changelog tracks the `rlmesh` Python package
on PyPI; the Rust crates are internal implementation detail and carry no separate user stability
promise.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). RLMesh is pre-1.0, so a
minor (`0.x`) release may include breaking changes, always called out under a Breaking heading with
a migration note. See the versioning policy for details.

## [0.1.0-rc.1] - 2026-06-16

First release candidate for 0.1.0. RLMesh connects models to environments across process,
dependency, and machine boundaries with a Gymnasium-style API.

### Added

- Serve Gymnasium-style environments and drive them with `reset`, `step`, `render`, and `close` over
  local or remote gRPC transports.
- DLPack-native `Tensor` transport with zero-copy NumPy, Torch, and JAX backends (#3).
- Run served environments locally or rebuild them identically in an isolated sandbox (`SandboxEnv`)
  (#8).
- Evaluate models locally, against a remote server, or inside a sandbox (`Model`, `RemoteModel`,
  `SandboxModel`) (#11).
- Tag-driven IO adapters that resolve environment tags against model specs at runtime (#9).
- Negotiated workflow editions content-pinned to the `2026.06` edition spec (#2).
- Per-lane `NEXT_STEP` autoreset contract for vector environments (#7).

### Changed

- Hardened the public Python API, space wrappers, and transport for the stable surface (#5).

[0.1.0-rc.1]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/v0.1.0-rc.1
