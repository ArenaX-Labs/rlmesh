# rlmesh-grpc

gRPC clients, servers, and wire helpers for RLMesh model-environment protocols.

Most applications should depend on the higher-level `rlmesh` crate. Use this crate to integrate at
the transport layer or to reach the gRPC environment and model services directly.

## Installation

```toml
[dependencies]
rlmesh-grpc = "0.1.0-rc.1"
```

## Status

Internal implementation detail of RLMesh, with no stability promise and no plan to stabilize it.
Build on the `rlmesh` Python package instead; see the
[compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-grpc
- Higher-level SDK: https://crates.io/crates/rlmesh

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
