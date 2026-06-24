# RLMesh

RLMesh is a Python SDK for model-environment evaluation. It serves Gymnasium-style environments, connects clients over local or remote transports, and adapts values for plain Python, NumPy, Torch, and JAX users.

> Pre-1.0 (`0.x`): the stable API may change in a minor release, with a migration note, so pin a minor range for active projects.

## Installation

Install from PyPI:

```bash
pip install rlmesh
```

Install optional adapters as needed:

```bash
pip install "rlmesh[numpy]"
pip install "rlmesh[gymnasium]"
pip install "rlmesh[torch]"
pip install "rlmesh[jax]"
```

## Quickstart

Install RLMesh with Gymnasium support and the NumPy client adapter:

```bash
pip install "rlmesh[gymnasium,numpy]"
```

In one process, serve any Gymnasium-compatible environment:

```python
import gymnasium as gym
from rlmesh import EnvServer

env = gym.make("CartPole-v1")
EnvServer(env, "127.0.0.1:5555").serve()
```

In another process, connect to it as a remote environment:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
observation, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Links

- Homepage: https://rlmesh.dev
- Documentation: https://docs.rlmesh.dev
- Repository: https://github.com/ArenaX-Labs/rlmesh
- Examples: https://github.com/ArenaX-Labs/rlmesh/tree/main/examples/python
- Issues: https://github.com/ArenaX-Labs/rlmesh/issues
- Default contact: research@competesai.com

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/LICENSE-MIT).

Python wheels also include third-party notices in [THIRD_PARTY_NOTICES.md](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/THIRD_PARTY_NOTICES.md).
