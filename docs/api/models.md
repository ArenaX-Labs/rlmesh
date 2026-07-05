# Models

```{note}
This is the autodoc API reference. For the authoring guide see {doc}`../user-guide/models`, and
{doc}`../user-guide/models/reference` for the full prediction-corner contract.
```

Models wrap a Python prediction function and run it against an RLMesh environment endpoint. The framework backend controls how observations are decoded before `predict_fn` runs and how returned actions are encoded.

Reach for a concrete `Model` class below in the value type your prediction function wants. Authors implement `load()` plus exactly one of the four predict corners (`predict`, `predict_chunk`, `predict_batch`, or `predict_chunk_batch`); the runtime dispatches to whichever is defined.

## Base Model

```{eval-rst}
.. autoclass:: rlmesh._models.base.ModelBase
   :members:
   :show-inheritance:
```

## Concrete Models

Concrete backend model classes inherit `ModelBase` and only change value conversion:

| Class        | Import               | Observation type                          | Action encoding              |
| ------------ | -------------------- | ----------------------------------------- | ---------------------------- |
| Native model | `rlmesh.Model`       | RLMesh-native values and primitives       | RLMesh-native values         |
| NumPy model  | `rlmesh.numpy.Model` | NumPy arrays, primitives, and containers  | NumPy arrays and primitives  |
| Torch model  | `rlmesh.torch.Model` | Torch tensors, primitives, and containers | Torch tensors and primitives |
| JAX model    | `rlmesh.jax.Model`   | JAX arrays, primitives, and containers    | JAX arrays and primitives    |

See {doc}`backends` for backend helpers.

## Served and Sandboxed Models

A model does not have to run in your process. `RemoteModel` dials a policy that is already served on an endpoint; `SandboxModel` runs a prebuilt `image://` tag in its own container. Both bind to an environment through {func}`~rlmesh.session` and expose the same {class}`~rlmesh.Session` drive loop.

```{eval-rst}
.. autoclass:: rlmesh.RemoteModel
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.SandboxModel
   :members:
```

## Sandbox Configuration

`SandboxBuild` groups the build-from-source settings for a `gym://` / `hf://` env source; `SandboxRuntime` groups the `docker run` flags applied when a prebuilt container starts. See {doc}`../user-guide/sandbox` for where each option goes.

```{eval-rst}
.. autoclass:: rlmesh.SandboxBuild
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.SandboxRuntime
   :members:
```
