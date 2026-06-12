# rlmesh-grpc

gRPC clients, servers, and wire helpers for RLMesh model-environment protocols.

Most applications should depend on the higher-level `rlmesh` crate. Use this crate when you are
integrating at the transport layer or need direct access to the gRPC environment and model services.

## Installation

```toml
[dependencies]
rlmesh-grpc = "0.1.0-beta.2"
```

## Status

This implementation crate is published for protocol integrators. The API is still unstable during
beta.

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-grpc
- Higher-level SDK: https://crates.io/crates/rlmesh

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
