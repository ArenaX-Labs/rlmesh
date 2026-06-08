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
