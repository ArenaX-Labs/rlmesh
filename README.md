<div align="center">

# RLMesh

**Gymnasium-compatible infrastructure for model-environment evaluation.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/rlmesh/) [![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh) [![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://docs.rlmesh.dev) [![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

</div>

RLMesh connects models to environments for evaluation. The Python SDK serves Gymnasium-style environments and lets evaluators call `reset`, `step`, `render`, and `close` over local or remote transports. Rust crates provide the lower-level runtime, protocol, and packaging layers.

## Project Status

RLMesh is in the **0.1.0 release-candidate** phase: the `2026.06` workflow edition is still provisional and seals at the final 0.1.0. The PyPI and crates.io badges above show the latest published version.

RLMesh is released and pre-1.0 (`0.x`). The Python package is the supported surface; a minor release may change a stable API with a migration note, so pin a minor range for active projects. The `rlmesh` facade crate and the CLI commands are the Rust-side surfaces we aim to stabilize; the other crates are internal implementation detail with no stability promise. See the [compatibility](https://docs.rlmesh.dev/compatibility/) and [versioning](https://docs.rlmesh.dev/versioning/) policies.

RLMesh is built around a language-neutral model-environment boundary. Python and Rust are supported today. Additional language bindings are future work, not part of the current public surface.

## Installation

Install the Python package from PyPI:

```bash
pip install rlmesh
```

## Quickstart

Install the Python package with Gymnasium support and the NumPy client adapter:

```bash
pip install "rlmesh[gymnasium,numpy]"
```

In one process, serve a standard Gymnasium environment:

```python
import gymnasium as gym
import rlmesh

env = gym.make("CartPole-v1")
rlmesh.EnvServer(env, "127.0.0.1:5555").serve()
```

In another process, connect to it as a remote environment:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

For runnable files and exact commands, see [`examples/python`](examples/python). Start with the quickstart, then try sandbox examples for Docker-backed environments or the optional MuJoCo and Pygame examples for heavier dependency stacks.

## Building the Rust SDK

The gRPC crates generate their stubs from `.proto` files at build time. Building any of them (`rlmesh`, `rlmesh-grpc`, `rlmesh-runtime`, `rlmesh-sandbox`) from source, including a plain `cargo add rlmesh && cargo build` from crates.io, requires the Protocol Buffers compiler `protoc` on the system. Install it from your package manager (for example `apt install protobuf-compiler` or `brew install protobuf`), or point `PROTOC` at an existing binary. The Python package has no such requirement; its wheels ship pre-built.

## Packages

- Python package: [`rlmesh`](python/rlmesh/README.md)
- Rust SDK: [`crates/rlmesh`](crates/rlmesh/README.md)
- CLI: [`rlmesh-cli`](crates/rlmesh-cli/README.md)
- Supporting crates: [`rlmesh-spaces`](crates/rlmesh-spaces/README.md), [`rlmesh-proto`](crates/rlmesh-proto/README.md), [`rlmesh-grpc`](crates/rlmesh-grpc/README.md), [`rlmesh-runtime`](crates/rlmesh-runtime/README.md), and [`rlmesh-sandbox`](crates/rlmesh-sandbox/README.md)

## Resources

- Documentation: https://docs.rlmesh.dev
- Examples: [`examples/python`](examples/python)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Local development: [`docs/local-dev.md`](docs/local-dev.md)
- Testing: [`docs/testing.md`](docs/testing.md)
- Compatibility: [`docs/compatibility.md`](docs/compatibility.md)
- Release process: [`docs/release.md`](docs/release.md)

## Contributing

Issues and focused pull requests are welcome. Larger API, architecture, or roadmap changes should start with an issue first. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

RLMesh is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
