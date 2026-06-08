# rlmesh-sandbox

Docker-backed environment packaging and execution for RLMesh evaluation workflows.

This crate contains the Rust sandbox primitives used to resolve environment sources, build container
specs, and launch packaged environments. Python users normally access sandboxed environments through
the `rlmesh` Python package.

## Installation

```toml
[dependencies]
rlmesh-sandbox = "0.1.0-beta.1"
```

## Status

Sandboxing is a supported public capability, but the direct Rust API is still unstable during beta.

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-sandbox
- Higher-level SDK: https://crates.io/crates/rlmesh

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
