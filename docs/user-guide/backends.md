# Framework Backends

A backend decides what RLMesh decodes values into at the Python boundary. The wire stays framework-neutral, so the backend is a client-side choice: it changes the type you receive from `reset` and `step`, not the protocol or the server. The same served environment can feed a NumPy client, a Torch client, and a JAX client at once.

Every backend exposes the same surface (`RemoteEnv`, `RemoteVectorEnv`, `Model`, and the sandbox sessions) under its own import path. Pick the one whose import matches the values your code already speaks.

```python
import rlmesh                       # plain Python and RLMesh-native values
from rlmesh.numpy import RemoteEnv  # NumPy arrays
from rlmesh.torch import RemoteEnv  # Torch tensors  (experimental)
from rlmesh.jax import RemoteEnv    # JAX arrays     (experimental)
```

## Choosing a backend

| Backend      | Import         | Install                       | Tensor leaves decode to | Status       | Reach for it when                                            |
| ------------ | -------------- | ----------------------------- | ----------------------- | ------------ | ------------------------------------------------------------ |
| Plain Python | `rlmesh`       | bundled                       | RLMesh-native values    | supported    | you want no array dependency, just primitives                |
| NumPy        | `rlmesh.numpy` | `pip install "rlmesh[numpy]"` | NumPy arrays            | supported    | examples, notebooks, and most CPU evaluation                 |
| Torch        | `rlmesh.torch` | `pip install "rlmesh[torch]"` | Torch tensors           | experimental | your model or environment already speaks Torch, GPU included |
| JAX          | `rlmesh.jax`   | `pip install "rlmesh[jax]"`   | JAX arrays (immutable)  | experimental | your pipeline is JAX                                         |

In every backend, only tensor leaves change type. Python primitives and nested containers (`dict`, `tuple`, lists) are preserved as they are.

## Plain Python

Top-level `rlmesh.RemoteEnv`, `rlmesh.RemoteVectorEnv`, and `rlmesh.Model` keep RLMesh-native values and Python primitives without requiring NumPy or Torch.

```python
import rlmesh

env = rlmesh.RemoteEnv("127.0.0.1:5555")
```

## NumPy

The NumPy backend decodes tensor leaves to NumPy arrays. It is the default choice for examples and notebooks.

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
```

```bash
pip install "rlmesh[numpy]"
```

The space wrappers from `env.observation_space` and `env.action_space` also use the NumPy backend, so `sample()` returns NumPy-compatible values where tensor leaves are involved.

## Torch

The Torch backend decodes tensor leaves to Torch tensors.

```python
from rlmesh.torch import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
```

```bash
pip install "rlmesh[torch]"
```

Decoding happens at the client boundary. A served environment can stay a plain Gymnasium environment and does not need to import Torch unless the environment itself does.

## JAX

The JAX backend decodes tensor leaves to JAX arrays, which are immutable by construction.

```python
from rlmesh.jax import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
```

```bash
pip install "rlmesh[jax]"
```

For conversion semantics and the supported JAX floor, see {doc}`../api/jax`.

## What "experimental" means here

Torch and JAX are device-bearing frameworks: their obs/action seam can carry tensors that live on a device, GPU included. NumPy and the plain backend have no device concept. That difference is the source of the limitations to know before you reach for them.

| Behavior                     | NumPy / plain                         | Torch / JAX                                                                                                                                                  |
| ---------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Device                       | none                                  | tensors can live on a device; an env-side action accepts `device=` (see {doc}`serving-environments`)                                                         |
| Serving a vector env         | `num_envs > 1` fans out via Gymnasium | not supported: Gymnasium vectorization concatenates observations with NumPy and discards framework tensors. Serve scalar (`num_envs=1`), or serve with NumPy |
| Mutability of decoded values | NumPy arrays are writable             | JAX arrays are immutable                                                                                                                                     |

The wire is framework-neutral regardless of backend, so a client's framework is independent of the server's. A NumPy environment can serve a Torch model client; nothing in between needs to agree on a framework.

## Models

Backends apply to model workers the same way. A `Model` from any backend hands `predict` values in that backend's types:

```python
from rlmesh.numpy import Model

model = Model(lambda obs: 0)
model.run("127.0.0.1:5555", max_episodes=1)
```

## Where next

- {doc}`remote-clients` — connect a client in the backend you chose.
- {doc}`serving-environments` — declare an env's action framework with `framework=` and `device=`.
- {doc}`evaluation` — run a model, whose `predict` receives backend-typed values.
- {doc}`adapters` — resolve a model's IO; adapter calls use the NumPy backend.
- {doc}`../api/numpy`, {doc}`../api/torch`, {doc}`../api/jax` — the per-backend autodoc.
