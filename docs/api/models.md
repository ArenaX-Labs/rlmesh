# Models

Model workers wrap a Python prediction function and run it against an RLMesh environment endpoint. The framework backend controls how observations are decoded before `predict_fn` runs and how returned actions are encoded.

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
