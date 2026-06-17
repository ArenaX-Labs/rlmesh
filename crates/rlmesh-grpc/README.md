# rlmesh-grpc

gRPC clients, servers, and wire helpers for RLMesh model-environment protocols.

Most applications should depend on the higher-level `rlmesh` crate. Use this crate when you are
integrating at the transport layer or need direct access to the gRPC environment and model services.

## Installation

```toml
[dependencies]
rlmesh-grpc = "0.1.0-rc.1"
```

## Status

Internal implementation crate. The Rust API is not stable yet and carries no compatibility promise
for now; stabilizing it is a near-term goal. Until then, build on the `rlmesh` Python package; see
the [compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-grpc
- Higher-level SDK: https://crates.io/crates/rlmesh

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
