# rlmesh

Rust crate for serving RLMesh environments and connecting Rust evaluators.

This is the Rust-side surface we intend to stabilize. It is not stable yet and carries no compatibility promise for now, but stabilizing it is a near-term goal. Most users should install the `rlmesh` Python package instead; reach for this crate to work directly against the Rust layer.

## Installation

```toml
[dependencies]
rlmesh = "0.1.0-rc.2"
```

## What it provides

- The `Env` and `SingleEnv` traits and the `SingleEnvAdapter` struct for environment bindings.
- `EnvServer` and `RemoteEnv` for serving and connecting to environments.
- `ModelWorker` for model-side workflows.
- Re-exported space, value, lifecycle, and error types used by the SDK.

## Examples

Two runnable examples mirror the Python quickstart. Serve an environment, then drive it from another process:

```bash
cargo run -p rlmesh --example serve_env    # host a CounterEnv on 127.0.0.1:5555
cargo run -p rlmesh --example run_model    # connect a model and run three episodes
```

The boundary is language-neutral, so the same server also accepts the Python client in `examples/python/quickstart/eval.py`, and a Rust `run_model` can drive a Python environment.

## Status

This is the Rust-side surface we intend to stabilize. It is not stable yet and carries no compatibility promise for now, but stabilizing it is a near-term goal. Until then, build on the `rlmesh` Python package; see the [compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh
- Python package: https://pypi.org/project/rlmesh/

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
