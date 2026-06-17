# rlmesh

Internal Rust crate for serving RLMesh environments and connecting Rust evaluators.

The Rust API is internal and not stable yet, with no compatibility promise for now; stabilizing it
is a near-term goal. Most users should install the `rlmesh` Python package instead; reach for this
crate only to work directly against the Rust layer.

## Installation

```toml
[dependencies]
rlmesh = "0.1.0-rc.1"
```

## What It Provides

- `Env`, `SingleEnv`, and `SingleEnvAdapter` traits for environment bindings.
- `EnvServer` and `RemoteEnv` for serving and connecting to environments.
- `ModelWorker` APIs for model-side workflows.
- Re-exported space, value, lifecycle, and error types used by the SDK.

## Examples

Two runnable examples mirror the Python quickstart. Serve an environment, then drive it from another
process:

```bash
cargo run -p rlmesh --example serve_env    # host a CounterEnv on 127.0.0.1:5555
cargo run -p rlmesh --example run_model    # connect a model and run three episodes
```

The boundary is language-neutral, so the same server also accepts the Python client in
`examples/python/quickstart/eval.py`, and a Rust `run_model` can drive a Python environment.

## Status

Internal implementation crate. The Rust API is not stable yet and carries no compatibility promise
for now; stabilizing it is a near-term goal. Until then, build on the `rlmesh` Python package; see
the [compatibility policy](https://docs.rlmesh.dev/compatibility/).

## Links

- Project: https://github.com/ArenaX-Labs/rlmesh
- Documentation: https://docs.rlmesh.dev
- API docs: https://docs.rs/rlmesh
- Python package: https://pypi.org/project/rlmesh/

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
