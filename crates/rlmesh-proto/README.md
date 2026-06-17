# rlmesh-proto

Protobuf definitions and generated gRPC stubs for RLMesh model-environment protocols.

Most users should depend on `rlmesh` or `rlmesh-grpc`. Use this crate when you need the raw Protobuf
message and service types for protocol-level integration.

## Installation

```toml
[dependencies]
rlmesh-proto = "0.1.0-rc.1"
```

## Build Requirements

The build script compiles the bundled `.proto` definitions with `tonic-prost-build`, which invokes
the Protocol Buffers compiler `protoc`. Building this crate, and therefore any crate that depends on
it (`rlmesh`, `rlmesh-grpc`, `rlmesh-runtime`, and `rlmesh-sandbox`), requires `protoc` on the
system. Install it from your package manager (for example `apt install protobuf-compiler` or
`brew install protobuf`), or point the `PROTOC` environment variable at an existing binary. The
proto3 `optional` fields used here need `protoc` 3.15 or newer. The `docs.rs` build image already
provides `protoc`, so the published API docs build without extra configuration.

## Status

Internal implementation detail of RLMesh, with no stability promise and no plan to stabilize it.
Build on the `rlmesh` Python package instead; see the
[compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh-proto
- Higher-level SDK: https://crates.io/crates/rlmesh

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
