# rlmesh-proto

Internal data types and workflow edition logic for the RLMesh run driver,
published only as a dependency of `rlmesh` and `rlmesh-cli`.

The driver still uses these protobuf messages in memory. Remote envs and models use
`rlmesh-realtime-v1` sessions; this crate defines no services or governed wire.

This crate is an internal component of `rlmesh`. The Rust API exposed here is unstable and will have frequent breaking changes.

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://rlmesh.dev/docs/
- API docs: https://docs.rs/rlmesh-proto
- Higher-level SDK: https://crates.io/crates/rlmesh

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
