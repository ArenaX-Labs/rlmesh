# NumPy

The NumPy adapter is the recommended default for examples, notebooks, and model code that already
works with arrays.

## What This Adapter Changes

`rlmesh.numpy` keeps the same environment, model, and sandbox behavior as the shared RLMesh client
APIs, but decodes tensor leaves to NumPy arrays. Space wrappers returned from NumPy clients also
sample NumPy-compatible values.

Install it with:

```bash
pip install --pre "rlmesh[numpy]"
```

| Concrete API                    | Shared behavior                        | Adapter-specific behavior                               |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| `rlmesh.numpy.RemoteEnv`        | {doc}`remote-envs` single clients      | Observations, actions, and render frames use arrays.    |
| `rlmesh.numpy.RemoteVectorEnv`  | {doc}`remote-envs` vector clients      | Batched values use NumPy-compatible containers.         |
| `rlmesh.numpy.Model`            | {doc}`models`                          | `predict_fn` receives NumPy-decoded observations.       |
| `rlmesh.numpy.SandboxEnv`       | {doc}`sandbox` single sandbox sessions | Owned sandbox client is `rlmesh.numpy.RemoteEnv`.       |
| `rlmesh.numpy.SandboxVectorEnv` | {doc}`sandbox` vector sandbox sessions | Owned sandbox client is `rlmesh.numpy.RemoteVectorEnv`. |

## Conversion Semantics

- `asarray(tensor)` returns a **zero-copy, read-only** view over the tensor bytes (NumPy enforces
  the read-only flag, unlike Torch). Call `.copy()` on the array if you need a writable buffer.
- `from_array(array)` always copies. It deliberately uses the buffer protocol rather than DLPack:
  read-only arrays cannot be exported over legacy DLPack, and decoded RLMesh views are read-only.
- `bfloat16` tensors have no buffer-protocol format, so `asarray` copies through raw bytes and needs
  the optional [ml_dtypes](https://github.com/jax-ml/ml_dtypes) package — install
  `rlmesh[bfloat16]`. Without it, `asarray` raises an `ImportError` naming that extra.

## Value Helpers

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

## RemoteEnv

```{eval-rst}
.. autoclass:: rlmesh.numpy.RemoteEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## RemoteVectorEnv

```{eval-rst}
.. autoclass:: rlmesh.numpy.RemoteVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Model

```{eval-rst}
.. autoclass:: rlmesh.numpy.Model
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Sandbox

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
