# RLMesh

![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)

RLMesh is evaluation infrastructure for RL and VLA systems. It provides Rust
and Python APIs for connecting models to environments, running local
model-environment workflows, and wrapping the same code for service-based
execution.

## Project Status

This repository is preparing for the `v0.1.0-beta1` OSS release. The beta is
intended for early adopters and feedback; APIs and package structure may still
change before a stable release.

## Packages

- Python package: `rlmesh` (`0.1.0b1`)
- Rust crates: `rlmesh`, `rlmesh-cli`, `rlmesh-grpc`, `rlmesh-proto`,
  `rlmesh-runtime`, `rlmesh-sandbox`, and `rlmesh-spaces`

## Installation

Python prerelease:

```bash
pip install --pre rlmesh
```

Install optional adapters as needed:

```bash
pip install --pre "rlmesh[numpy]"
pip install --pre "rlmesh[gymnasium]"
pip install --pre "rlmesh[torch]"
```

Rust prerelease:

```toml
rlmesh = "0.1.0-beta.1"
```

## Quickstart

Run a sampled-action eval against a tiny served environment:

```bash
pip install --pre "rlmesh[numpy]"
python examples/python/quickstart/serve.py
```

In a second terminal:

```bash
python examples/python/quickstart/eval.py
```

The eval process connects to whichever example `EnvServer` is running on
`127.0.0.1:5555`, samples actions from the advertised remote action space, and
runs one short episode. You can swap the first terminal for
`examples/python/sai-mujoco/serve.py` or `examples/python/sai-pygame/serve.py`
from those examples' own mise/uv environments.

## Development

Install local tools and development dependencies:

```bash
mise run setup
```

Run the fast local checks:

```bash
mise run check
mise run test
```

The static check task covers formatting, linting, Python type checking, Rust
library docs with warnings denied, proto linting, and generated native stub
drift. The test task runs the Rust workspace tests plus Python unit and
integration tests.

Run the public API and system harness checks used by CI:

```bash
mise run test:python:api-contract
mise run test:system:harness
mise run release:rust:package
```

## Installed-Artifact System Tests

RLMesh has a separate system runner for testing built Python wheels in clean
`uv` environments. It installs the wheel under test plus private fixture
entrypoints, then exercises process boundaries, optional dependencies,
deterministic traces, and artifact-level benchmark signal.

```bash
mise run test:system:list
mise run test:system -- --dry-run
mise run test:system
mise run test:system:heavy
```

System profiles:

- `basic`: fast process-boundary traces and NumPy artifact checks.
- `gymnasium`: real Gymnasium environment IDs.
- `torch`: Torch action and tensor artifact checks.
- `mujoco`: MuJoCo-backed Gymnasium environments.
- `heavy`: aggregate Gymnasium, Torch, and MuJoCo profile.

The local current-platform wheel builder may produce plain `linux_*` wheels for
smoke testing. Release wheels must use uploadable platform tags such as
`manylinux`, `musllinux`, `macosx`, or `win`.

## Local Release Gate

Before publishing the beta from a local machine, run:

```bash
mise run release:check
```

This runs static checks, Rust tests, Python unit/integration/API contract tests,
system runner tests, Cargo package verification, local wheel builds, and basic
plus heavy installed-artifact system validation. It does not publish anything.

Publishing is intentionally separate and manual for this beta. See
[docs/release.md](docs/release.md) for the maintainer release process.

## Contributing

Issues and focused pull requests are welcome. Larger API, architecture, or
roadmap changes should start with an issue first so the direction is clear
before implementation work begins.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license
