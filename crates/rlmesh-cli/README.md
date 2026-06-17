# rlmesh-cli

Command-line entrypoint for inspecting RLMesh distributions and running support commands.

This crate publishes the `rlmesh` binary. The CLI is small: `version` for distribution inspection,
plus internal viewer plumbing used by the Python package.

## Installation

```bash
cargo install rlmesh-cli --version 0.1.0-rc.1
```

## Commands

```bash
rlmesh version
```

## Status

The crate's Rust API is internal, with no stability promise. The CLI commands are a supported
surface we intend to stabilize; today only `version` is stable. Stabilizing the rest is a near-term
goal. See the [compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-cli

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
