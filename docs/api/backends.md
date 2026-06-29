# Framework Backends

```{note}
This is the autodoc API reference. For how backends decode values at the Python boundary see
{doc}`../user-guide/backends`.
```

Each backend keeps the same environment, model, and sandbox behavior as the shared RLMesh client APIs, but decodes tensor leaves to its own array type. Space wrappers returned from a backend's clients also sample values compatible with that type. The Torch and JAX backends are experimental.

## NumPy

Use the NumPy backend for examples, notebooks, and model code that already works with arrays. Install it with:

```bash
pip install "rlmesh[numpy]"
```

| Concrete API                    | Shared behavior                        | Backend-specific behavior                                |
| ------------------------------- | -------------------------------------- | -------------------------------------------------------- |
| `rlmesh.numpy.RemoteEnv`        | {doc}`serving` single clients          | Observations, actions, and render frames use arrays.     |
| `rlmesh.numpy.RemoteVectorEnv`  | {doc}`serving` vector clients          | Batched values use NumPy-compatible containers.          |
| `rlmesh.numpy.Model`            | {doc}`models`                          | `predict_fn` receives NumPy-decoded observations.        |
| `rlmesh.numpy.SandboxEnv`       | {doc}`sandbox` single sandbox sessions | Owned sandbox client is `rlmesh.numpy.RemoteEnv`.        |
| `rlmesh.numpy.SandboxModel`     | {doc}`sandbox`                         | Runs a model policy in its own container (experimental). |
| `rlmesh.numpy.SandboxVectorEnv` | {doc}`sandbox` vector sandbox sessions | Owned sandbox client is `rlmesh.numpy.RemoteVectorEnv`.  |

### Conversion Semantics

- `asarray(tensor)` returns a **writable copy** of the tensor bytes, matching Gymnasium where `reset`/`step` observations are writable (so `obs /= 255.0` works). For a zero-copy, read-only view that shares the tensor buffer, use `numpy.from_dlpack(tensor)` or the buffer protocol.
- `from_array(array)` always copies: it makes the array C-contiguous and serializes its bytes into a fresh RLMesh tensor.

### Value Helpers

```{eval-rst}
.. autofunction:: rlmesh.numpy.ensure_available
```

```{eval-rst}
.. autofunction:: rlmesh.numpy.asarray
```

```{eval-rst}
.. autofunction:: rlmesh.numpy.from_array
```

```{eval-rst}
.. autofunction:: rlmesh.numpy.space_from_spec
```

### RemoteEnv

```{eval-rst}
.. autoclass:: rlmesh.numpy.RemoteEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### RemoteVectorEnv

```{eval-rst}
.. autoclass:: rlmesh.numpy.RemoteVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### Model

```{eval-rst}
.. autoclass:: rlmesh.numpy.Model
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### Sandbox

```{eval-rst}
.. autoclass:: rlmesh.numpy.SandboxEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.numpy.SandboxVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Torch (experimental)

Use the Torch backend for model code that already works with Torch tensors, especially when you want a zero-copy view over the wire buffer. Install it with:

```bash
pip install "rlmesh[torch]"
```

| Concrete API                    | Shared behavior                        | Backend-specific behavior                               |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| `rlmesh.torch.RemoteEnv`        | {doc}`serving` single clients          | Observations, actions, and render frames use tensors.   |
| `rlmesh.torch.RemoteVectorEnv`  | {doc}`serving` vector clients          | Batched values use Torch-compatible containers.         |
| `rlmesh.torch.Model`            | {doc}`models`                          | `predict_fn` receives Torch-decoded observations.       |
| `rlmesh.torch.SandboxEnv`       | {doc}`sandbox` single sandbox sessions | Owned sandbox client is `rlmesh.torch.RemoteEnv`.       |
| `rlmesh.torch.SandboxVectorEnv` | {doc}`sandbox` vector sandbox sessions | Owned sandbox client is `rlmesh.torch.RemoteVectorEnv`. |

### Memory Sharing and Mutation

Decoded observations are owned, writable copies, so `predict_fn` can normalize in place (`img.div_(255)`) without corrupting the wire buffer. `as_tensor(tensor)` is the zero-copy opt-in: the Torch tensor shares memory with the RLMesh tensor over DLPack.

```{warning}
A zero-copy `as_tensor(tensor)` view shares memory. RLMesh flags the export read-only, but Torch,
like most DLPack consumers, does not enforce that flag, so an in-place write corrupts the RLMesh
tensor for every other view of the same data, including NumPy views in the same process. Treat a
shared view as read-only, or pass `copy=True`.
```

Conversion details:

- Decode uses `torch.utils.dlpack.from_dlpack`; `bool` tensors fall back to a buffer copy on Torch older than 2.2 (no bool DLPack support there).
- `uint16`, `uint32`, and `uint64` dtypes require Torch 2.3 or newer.
- Encode (`from_tensor`) detaches, moves to CPU, and exports over DLPack; NumPy is not required.

### Value Helpers

```{eval-rst}
.. autofunction:: rlmesh.torch.ensure_available
```

```{eval-rst}
.. autofunction:: rlmesh.torch.as_tensor
```

```{eval-rst}
.. autofunction:: rlmesh.torch.from_tensor
```

```{eval-rst}
.. autofunction:: rlmesh.torch.space_from_spec
```

### RemoteEnv

```{eval-rst}
.. autoclass:: rlmesh.torch.RemoteEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### RemoteVectorEnv

```{eval-rst}
.. autoclass:: rlmesh.torch.RemoteVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### Model

```{eval-rst}
.. autoclass:: rlmesh.torch.Model
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### Sandbox

```{eval-rst}
.. autoclass:: rlmesh.torch.SandboxEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.torch.SandboxVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## JAX (experimental)

Use the JAX backend for model code that already works with JAX arrays; decoded arrays are immutable, so there is no mutation hazard. Install it with:

```bash
pip install "rlmesh[jax]"
```

| Concrete API                  | Shared behavior                        | Backend-specific behavior                             |
| ----------------------------- | -------------------------------------- | ----------------------------------------------------- |
| `rlmesh.jax.RemoteEnv`        | {doc}`serving` single clients          | Observations, actions, and render frames use arrays.  |
| `rlmesh.jax.RemoteVectorEnv`  | {doc}`serving` vector clients          | Batched values use JAX-compatible containers.         |
| `rlmesh.jax.Model`            | {doc}`models`                          | `predict_fn` receives JAX-decoded observations.       |
| `rlmesh.jax.SandboxEnv`       | {doc}`sandbox` single sandbox sessions | Owned sandbox client is `rlmesh.jax.RemoteEnv`.       |
| `rlmesh.jax.SandboxVectorEnv` | {doc}`sandbox` vector sandbox sessions | Owned sandbox client is `rlmesh.jax.RemoteVectorEnv`. |

### Conversion Semantics

- `asarray(tensor)` imports over DLPack. XLA shares RLMesh's 64-byte-aligned buffers zero-copy and copies otherwise; JAX arrays are immutable either way, so there is no mutation hazard.
- `from_array(array)` moves the array to CPU if needed, blocks until ready, and copies the elements into a fresh RLMesh tensor.
- `int64`, `uint64`, and `float64` values require JAX 64-bit mode (`jax.config.update("jax_enable_x64", True)`); without it JAX itself demotes those dtypes.
- Requires `jax >= 0.4.24`, the first release with DLPack `bool` support. `ensure_available` enforces the floor at runtime.

### Value Helpers

```{eval-rst}
.. autofunction:: rlmesh.jax.ensure_available
```

```{eval-rst}
.. autofunction:: rlmesh.jax.asarray
```

```{eval-rst}
.. autofunction:: rlmesh.jax.from_array
```

```{eval-rst}
.. autofunction:: rlmesh.jax.space_from_spec
```

### RemoteEnv

```{eval-rst}
.. autoclass:: rlmesh.jax.RemoteEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### RemoteVectorEnv

```{eval-rst}
.. autoclass:: rlmesh.jax.RemoteVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### Model

```{eval-rst}
.. autoclass:: rlmesh.jax.Model
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

### Sandbox

```{eval-rst}
.. autoclass:: rlmesh.jax.SandboxEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.jax.SandboxVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Where next

- {doc}`../user-guide/backends` for how decoding works at the Python boundary.
- {doc}`serving`, {doc}`models`, and {doc}`sandbox` for the shared client behavior each backend reuses.
