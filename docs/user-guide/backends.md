# Framework Backends

Framework backends control how values are decoded at the Python boundary.

## Plain Python

Top-level `rlmesh.RemoteEnv`, `rlmesh.RemoteVectorEnv`, and `rlmesh.Model` preserve RLMesh-native values and Python primitives without requiring NumPy or Torch.

```python
import rlmesh

env = rlmesh.RemoteEnv("127.0.0.1:5555")
```

## NumPy

Use NumPy for examples and notebooks. Tensor leaves decode to NumPy arrays, while Python primitives and nested containers are preserved.

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
```

Install it with:

```bash
pip install "rlmesh[numpy]"
```

The space wrappers returned by `env.observation_space` and `env.action_space` also use the NumPy backend, so `sample()` returns NumPy-compatible values where tensor leaves are involved.

## Torch

The Torch backend is experimental. Tensor leaves decode to Torch tensors.

```python
from rlmesh.torch import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
```

Install it with:

```bash
pip install "rlmesh[torch]"
```

Torch decoding happens at the client boundary. The server can remain a normal Gymnasium environment and does not need to import Torch unless the environment itself needs it.

## JAX

The JAX backend is experimental. Tensor leaves decode to JAX arrays, which are immutable by construction.

```python
from rlmesh.jax import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
```

Install it with:

```bash
pip install "rlmesh[jax]"
```

For conversion semantics and the supported JAX floor, see {doc}`../api/jax`.

## Models

Backends also apply to model workers:

```python
from rlmesh.numpy import Model

model = Model(lambda obs: 0)
model.run("127.0.0.1:5555", max_episodes=1)
```
