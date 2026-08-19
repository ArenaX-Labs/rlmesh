# rlmesh-cli

Command-line entrypoint for RLMesh authentication, profile management, registry access, and distribution inspection.

This crate publishes the `rlmesh` binary and is also embedded in the Python package as the `python -m rlmesh` entrypoint. Named profiles keep platform configuration separate, while credentials are stored in the operating system keychain when one is available.

## Installation

```bash
cargo install rlmesh-cli --version 0.1.0
```

## Commands

```bash
# Sign in with the browser-based device flow
rlmesh login

# Inspect the current session
rlmesh whoami

# Authenticate Docker with the platform registry
rlmesh registry login

# List and switch between platform profiles
rlmesh profile list
rlmesh profile use <name>

# Inspect the installed CLI distribution
rlmesh version
```

Run `rlmesh --help` or `rlmesh <command> --help` for the complete command reference.

## Status

The crate's Rust API is internal, with no stability promise. The CLI command surface is in beta. See the [compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-cli

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
