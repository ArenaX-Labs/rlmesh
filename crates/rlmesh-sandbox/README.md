# rlmesh-sandbox

Docker-backed environment packaging and execution for RLMesh environments.

This crate holds the Rust sandbox primitives that resolve environment sources, build container specs, and launch packaged environments. Python users normally reach sandboxed environments through the `rlmesh` Python package.

## Installation

```toml
[dependencies]
rlmesh-sandbox = "0.1.0-rc.1"
```

## Status

Internal implementation detail of RLMesh, with no stability promise and no plan to stabilize it. Build on the `rlmesh` Python package instead; see the [compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-sandbox
- Higher-level SDK: https://crates.io/crates/rlmesh

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
