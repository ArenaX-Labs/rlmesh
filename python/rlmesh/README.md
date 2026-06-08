# RLMesh

RLMesh is a Python SDK for model-environment evaluation workflows. It can serve Gymnasium-style
environments, connect clients over local or remote transports, and adapt values for plain Python,
NumPy, and Torch users.

> Beta: APIs and package structure may change before the stable release.

## Installation

Install the published beta from PyPI:

```bash
pip install --pre rlmesh
```

Install optional adapters as needed:

```bash
pip install --pre "rlmesh[numpy]"
pip install --pre "rlmesh[gymnasium]"
pip install --pre "rlmesh[torch]"
```

## Quickstart

Install RLMesh with Gymnasium support:

```bash
pip install --pre "rlmesh[gymnasium]"
```

In one process, serve any Gymnasium-compatible environment:

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
observation, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Runnable examples and exact commands live in the repository under `examples/python`.

## Links

- Homepage: https://rlmesh.dev
- Documentation: https://docs.rlmesh.dev
- Repository: https://github.com/ArenaX-Labs/rlmesh
- Examples: https://github.com/ArenaX-Labs/rlmesh/tree/main/examples/python
- Issues: https://github.com/ArenaX-Labs/rlmesh/issues
- Default contact: research@competesai.com

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See
[LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/LICENSE-APACHE) and
[LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/LICENSE-MIT).

Python wheels also include third-party notices in
[THIRD_PARTY_NOTICES.md](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/THIRD_PARTY_NOTICES.md).
