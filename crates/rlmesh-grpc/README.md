# rlmesh-grpc

gRPC clients, servers, and wire helpers for RLMesh model-environment protocols.

Most applications should depend on the higher-level `rlmesh` crate. Use this crate when you are
integrating at the transport layer or need direct access to the gRPC environment and model services.

## Installation

```toml
[dependencies]
rlmesh-grpc = "0.1.0-beta.1"
```

## Status

This is a public repository implementation crate. It is published so protocol integrators can build
against it, but the API is still unstable during beta.

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-grpc
- Higher-level SDK: https://crates.io/crates/rlmesh
