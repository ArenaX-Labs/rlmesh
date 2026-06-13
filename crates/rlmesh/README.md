# rlmesh

Rust SDK for serving RLMesh environments and connecting Rust evaluators.

Use this crate when you want to serve environments, connect to remote environments, or build
directly against RLMesh's Rust API. Most Python users should install the `rlmesh` Python package
instead.

## Installation

```toml
[dependencies]
rlmesh = "0.1.0-beta.2"
```

## What It Provides

- `Env`, `SingleEnv`, and `SingleEnvAdapter` traits for environment bindings.
- `EnvServer` and `RemoteEnv` for serving and connecting to environments.
- `ModelWorker` APIs for model-side workflows.
- Re-exported space, value, lifecycle, and error types used by the SDK.

## Status

This crate is part of the `0.1.0-beta.2` release line. The public Rust API is supported for beta
users and may still change before a stable release.

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh
- Python package: https://pypi.org/project/rlmesh/

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
