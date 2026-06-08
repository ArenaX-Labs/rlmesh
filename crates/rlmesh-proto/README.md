# rlmesh-proto

Protobuf definitions and generated gRPC stubs for RLMesh model-environment protocols.

Most users should depend on `rlmesh` or `rlmesh-grpc`. Use this crate when you need the raw protobuf
message and service types for protocol-level integration.

## Installation

```toml
[dependencies]
rlmesh-proto = "0.1.0-beta.1"
```

## Status

This supporting crate is part of the supported beta protocol surface. Message shape may still change
before the stable release.

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-proto
- Higher-level SDK: https://crates.io/crates/rlmesh
