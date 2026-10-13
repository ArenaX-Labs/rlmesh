<div align="center">

<img src="docs/assets/brand-mark.svg" alt="RLMesh" width="96" height="96">

# RLMesh

**Plug any model into any Gymnasium environment, locally or over the network.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/) [![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh) [![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://rlmesh.dev/docs/)

</div>

Serve an environment in one process, drive it from another. Same `reset` / `step` / `render` / `close` you already know, just over the wire. Python on top, Rust underneath.

## Install

```bash
pip install "rlmesh[gymnasium,numpy]"
```

## Quickstart

Serve an env:

```python
import gymnasium as gym
import rlmesh

rlmesh.EnvServer(gym.make("CartPole-v1"), "127.0.0.1:5555").serve()
```

Use it from somewhere else:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)

done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated

env.close()
```

More in [`examples/python`](examples/python): adapters, Docker sandboxes, bring-your-own containers.

## Rust

Grab the [`rlmesh`](crates/rlmesh/README.md) crate. You'll need `protoc` installed (`apt install protobuf-compiler` / `brew install protobuf`) since the gRPC stubs are generated at build time.

## Status

Pre-1.0. The Python package is the stable-ish surface; pin a minor version if you depend on it. See [compatibility](https://rlmesh.dev/docs/releases/compatibility/) for the details.

## Links

- Docs: https://rlmesh.dev/docs/
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · [local dev](docs/local-dev.md) · [testing](docs/testing.md)

## License

MIT or Apache-2.0, your pick. JPEG encoding is based in part on the work of the Independent JPEG Group.
