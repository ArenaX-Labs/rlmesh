# JAX

The JAX backend is experimental in this beta.

## What This Backend Changes

`rlmesh.jax` keeps the same environment, model, and sandbox behavior as the shared RLMesh client
APIs, but decodes tensor leaves to JAX arrays. Space wrappers returned from JAX clients also sample
JAX-compatible values.

Install it with:

```bash
pip install --pre "rlmesh[jax]"
```

| Concrete API                  | Shared behavior                        | Backend-specific behavior                             |
| ----------------------------- | -------------------------------------- | ----------------------------------------------------- |
| `rlmesh.jax.RemoteEnv`        | {doc}`remote-envs` single clients      | Observations, actions, and render frames use arrays.  |
| `rlmesh.jax.RemoteVectorEnv`  | {doc}`remote-envs` vector clients      | Batched values use JAX-compatible containers.         |
| `rlmesh.jax.Model`            | {doc}`models`                          | `predict_fn` receives JAX-decoded observations.       |
| `rlmesh.jax.SandboxEnv`       | {doc}`sandbox` single sandbox sessions | Owned sandbox client is `rlmesh.jax.RemoteEnv`.       |
| `rlmesh.jax.SandboxVectorEnv` | {doc}`sandbox` vector sandbox sessions | Owned sandbox client is `rlmesh.jax.RemoteVectorEnv`. |

## Conversion Semantics

- `asarray(tensor)` imports over DLPack. XLA shares RLMesh's 64-byte-aligned buffers zero-copy and
  copies otherwise; JAX arrays are immutable either way, so there is no mutation hazard.
- `from_array(array)` moves the array to CPU if needed, blocks until ready, and copies the elements
  into a fresh RLMesh tensor.
- `int64`, `uint64`, and `float64` values require JAX 64-bit mode
  (`jax.config.update("jax_enable_x64", True)`); without it JAX itself demotes those dtypes.
- Requires `jax >= 0.4.24`, the first release with DLPack `bool` support. `ensure_available`
  enforces the floor at runtime.

## Value Helpers

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

## RemoteEnv

```{eval-rst}
.. autoclass:: rlmesh.jax.RemoteEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## RemoteVectorEnv

```{eval-rst}
.. autoclass:: rlmesh.jax.RemoteVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Model

```{eval-rst}
.. autoclass:: rlmesh.jax.Model
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Sandbox

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
