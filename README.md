<div align="center">

# RLMesh

**Gymnasium-compatible infrastructure for model-environment evaluation.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/rlmesh/)
[![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh)
[![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://docs.rlmesh.dev)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

</div>

RLMesh is an evaluation framework for connecting models to environments. The Python SDK serves
Gymnasium-style environments, connects local or remote evaluators, and keeps the same workflow
usable when evaluation moves behind a service boundary. Rust crates provide the lower-level runtime,
protocol, and packaging layers.

## Project Status

RLMesh is currently in beta. The published beta is intended for early adopters and feedback; APIs
and package structure may still change before a stable release.

RLMesh is designed around a language-neutral model-environment boundary. Python and Rust are the
current supported surfaces, and the project intends to support clean, simple bindings for additional
languages where there is demand, with C++ as a likely early candidate.

## Installation

Install the published Python beta from PyPI:

```bash
pip install --pre rlmesh
```

## Quickstart

Install the Python package with Gymnasium support and the NumPy client adapter:

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
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

For runnable files and exact commands, see the [`examples/python`](examples/python) index. Start
with the quickstart example, use sandbox examples for owned Docker-backed environments, and use the
optional MuJoCo and Pygame examples for isolated heavier dependency stacks.

## Building the Rust SDK

The Rust crates generate their gRPC stubs from `.proto` files at build time, so building any of them
(`rlmesh`, `rlmesh-grpc`, `rlmesh-runtime`, `rlmesh-sandbox`, `rlmesh-cli`) from source — including
a plain `cargo add rlmesh && cargo build` from crates.io — requires the Protocol Buffers compiler
`protoc` on the system. Install it from your package manager (for example
`apt install protobuf-compiler` or `brew install protobuf`), or point `PROTOC` at an existing
binary. The Python package has no such requirement; its wheels ship pre-built.

## Packages

- Python package: [`rlmesh`](python/rlmesh/README.md)
- Rust SDK: [`crates/rlmesh`](crates/rlmesh/README.md)
- CLI: [`rlmesh-cli`](crates/rlmesh-cli/README.md)
- Supporting crates: [`rlmesh-spaces`](crates/rlmesh-spaces/README.md),
  [`rlmesh-proto`](crates/rlmesh-proto/README.md), [`rlmesh-grpc`](crates/rlmesh-grpc/README.md),
  [`rlmesh-runtime`](crates/rlmesh-runtime/README.md), and
  [`rlmesh-sandbox`](crates/rlmesh-sandbox/README.md)

## Resources

- Documentation: https://docs.rlmesh.dev
- Examples: [`examples/python`](examples/python)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Local development: [`docs/local-dev.md`](docs/local-dev.md)
- Testing: [`docs/testing.md`](docs/testing.md)
- Compatibility: [`docs/compatibility.md`](docs/compatibility.md)
- Release process: [`docs/release.md`](docs/release.md)

## Contributing

Issues and focused pull requests are welcome. Larger API, architecture, or roadmap changes should
start with an issue first. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

RLMesh is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
