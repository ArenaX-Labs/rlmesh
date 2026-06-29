# Models

```{note}
This is the autodoc API reference. For the authoring guide see {doc}`../user-guide/models`, and
{doc}`../user-guide/models/reference` for the full prediction-corner contract.
```

Model workers wrap a Python prediction function and run it against an RLMesh environment endpoint. The framework backend controls how observations are decoded before `predict_fn` runs and how returned actions are encoded.

Reach for a concrete `Model` class below in the value type your prediction function wants. Authors implement `load()` plus exactly one of the four prediction corners (`predict`, `predict_chunk`, `predict_batch`, or `predict_chunk_batch`); the runtime dispatches to whichever is defined.

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

See {doc}`numpy`, {doc}`torch`, and {doc}`jax` for backend helpers.
