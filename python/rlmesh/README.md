# RLMesh

RLMesh is a Python SDK that connects models to environments and robots. It serves Gymnasium-style environments and drives them from another process, runs real-time sessions between robots and policies on a network (`rlmesh.realtime`, Experimental), and adapts values for plain Python, NumPy, Torch, and JAX users.

> **0.2 is an alpha.** It replaces the 0.1 gRPC protocol with real-time sessions; 0.1 and 0.2 do not talk to each other, and anything may change before 0.2.0. Install it with `pip install --pre rlmesh`, or pin `rlmesh==0.2.0a1`. A plain `pip install rlmesh` gives 0.1.

## Quickstart

Install RLMesh with Gymnasium support and the NumPy client adapter:

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
```

In one process, serve any Gymnasium-compatible environment and print its address:

```python
import gymnasium as gym
from rlmesh import EnvServer

server = EnvServer(gym.make("CartPole-v1"), "127.0.0.1:5555/cartpole")
print(server.address)  # 127.0.0.1:5555#<key>/cartpole
server.serve()
```

In another process, connect to the address it printed:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555#<key>/cartpole")  # paste the printed address
observation, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

The part after `#` is a key the server makes when it starts; another process needs it to connect, so copy the whole address each time you start the server.

## Installation

Install only the extras you need (`--pre` picks the 0.2 alpha):

```bash
pip install --pre rlmesh
pip install --pre "rlmesh[numpy]"
pip install --pre "rlmesh[gymnasium]"
pip install --pre "rlmesh[torch]"
pip install --pre "rlmesh[jax]"
```

## Links

- Homepage: https://rlmesh.dev
- Documentation: https://rlmesh.dev/docs/
- Repository: https://github.com/ArenaX-Labs/rlmesh
- Examples: https://github.com/ArenaX-Labs/rlmesh/tree/main/examples/python
- Issues: https://github.com/ArenaX-Labs/rlmesh/issues
- Default contact: research@competesai.com

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/LICENSE-MIT).

Python wheels also include third-party notices in [THIRD_PARTY_NOTICES.md](https://github.com/ArenaX-Labs/rlmesh/blob/main/python/rlmesh/THIRD_PARTY_NOTICES.md).
